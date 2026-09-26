"""
表格 OCR 提取工作流
文档图片 → 表格检测 → 结构识别 → 文字识别 → HTML/CSV/JSON

用法:
    python table_ocr_pipeline.py --input document.jpg --output output/
    python table_ocr_pipeline.py --input ./images/ --output output/
"""
import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# ==================== CRNN 模型定义（来自项目1） ====================

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-/()&$:;#@!?'\"+=%*[]{}\\|<>~`^ "
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d((2, 1), (2, 1))
        self.conv7 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn7 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        x = F.relu(self.bn7(self.conv7(x)))
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True, dropout=0.3)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent


class CRNN(nn.Module):
    """CRNN: CNN + BiLSTM + CTC"""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.cnn = CNN()
        self.rnn = RNN(512, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        conv_feat = self.cnn(x)
        b, c, h, w = conv_feat.size()
        conv_feat = conv_feat.squeeze(2).permute(0, 2, 1)
        rnn_feat = self.rnn(conv_feat)
        output = self.fc(rnn_feat)
        output = F.log_softmax(output, dim=2)
        return output


# ==================== CTC 解码 ====================

def ctc_greedy_decode(output):
    """贪婪解码 CTC 输出为文本"""
    _, max_indices = output.max(dim=2)
    texts = []
    for i in range(max_indices.size(0)):
        indices = max_indices[i].cpu().numpy()
        text = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx and idx != 0:
                text.append(IDX_TO_CHAR.get(idx, ''))
            prev_idx = idx
        texts.append(''.join(text))
    return texts


# ==================== 工作流主类 ====================

class TableOCRPipeline:
    """
    表格 OCR 提取流水线
    1. 检测表格区域（Faster R-CNN）
    2. 识别表格结构（Faster R-CNN, cell 检测）
    3. 识别文字（CRNN + CTC）
    4. 组装为结构化表格
    """

    def __init__(self, detector_path, structure_path, ocr_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from model import TableDetector
        from utils import load_checkpoint

        # Step 1: 表格检测模型
        self.detector = TableDetector(num_classes=2, pretrained=False).to(self.device)
        load_checkpoint(self.detector, None, detector_path)
        self.detector.eval()
        print(f"  [OK] Detector: {detector_path}")

        # Step 2: 结构识别模型
        self.structure_model = TableDetector(
            num_classes=3, pretrained=False,
            detections_per_img=2000, nms_thresh=0.3
        ).to(self.device)
        load_checkpoint(self.structure_model, None, structure_path)
        self.structure_model.eval()
        print(f"  [OK] Structure: {structure_path}")

        # Step 3: CRNN 文字识别模型
        self.ocr_model = CRNN().to(self.device)
        checkpoint = torch.load(ocr_path, map_location='cpu', weights_only=False)
        self.ocr_model.load_state_dict(checkpoint['model_state_dict'])
        self.ocr_model.eval()
        print(f"  [OK] OCR: {ocr_path}")

        # 预处理
        self.transform = T.Compose([
            T.Resize([1024, 1024]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ---------- Step 1: 表格检测 ----------

    def _detect_tables(self, image):
        """检测文档中的表格区域"""
        orig_w, orig_h = image.size
        img_tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            outputs = self.detector([img_tensor])

        output = outputs[0]
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()

        # 过滤 + NMS
        keep = scores > 0.5
        boxes = boxes[keep]
        scores = scores[keep]

        if len(boxes) > 0:
            from torchvision.ops import nms
            nms_keep = nms(boxes, scores, 0.5)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]

        # 映射回原图坐标
        scale_x = orig_w / 1024
        scale_y = orig_h / 1024
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        tables = []
        for i in range(len(boxes)):
            tables.append({
                'bbox': boxes[i].tolist(),
                'score': scores[i].item(),
            })
        return tables

    # ---------- Step 2: 结构识别 ----------

    def _recognize_structure(self, table_image):
        """识别表格内的 cell"""
        orig_w, orig_h = table_image.size
        img_tensor = self.transform(table_image).to(self.device)

        with torch.no_grad():
            outputs = self.structure_model([img_tensor])

        output = outputs[0]
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        labels = output['labels'].cpu()

        # 过滤 + NMS
        keep = scores > 0.3
        boxes = boxes[keep]
        scores = scores[keep]

        if len(boxes) > 0:
            from torchvision.ops import nms
            nms_keep = nms(boxes, scores, 0.3)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]

        # 映射回原图坐标
        scale_x = orig_w / 1024
        scale_y = orig_h / 1024
        if len(boxes) > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

        cells = []
        for i in range(len(boxes)):
            cells.append({
                'bbox': boxes[i].tolist(),
                'score': scores[i].item(),
            })
        return cells

    # ---------- Step 3: 文字识别 ----------

    def _recognize_text(self, cell_image):
        """
        识别单个 cell 的文字
        支持多行：水平投影切行 → 逐行识别 → 拼接
        """
        # 转为 numpy
        img_np = np.array(cell_image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        h, w = gray.shape

        # 太小的 cell 直接跳过
        if h < 5 or w < 5:
            return ''

        # 检测是否多行（高度 > 宽度的 0.3 倍，且高度 > 50）
        lines = self._split_lines(gray)

        results = []
        for line_img in lines:
            text = self._ocr_single_line(line_img)
            if text.strip():
                results.append(text.strip())

        return '\n'.join(results) if results else ''

    def _split_lines(self, gray_img):
        """水平投影切行"""
        h, w = gray_img.shape

        # 高度太小，直接当单行
        if h < 40:
            return [gray_img]

        # 二值化（文字为白色）
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 水平投影
        projection = np.sum(binary, axis=1) / 255

        # 找文字区域（投影值 > 阈值）
        threshold = w * 0.05  # 至少 5% 的像素是文字
        text_rows = projection > threshold

        # 找切分点（文字区域之间的间隙）
        lines = []
        in_text = False
        start = 0

        for i in range(h):
            if text_rows[i] and not in_text:
                start = i
                in_text = True
            elif not text_rows[i] and in_text:
                if i - start > 10:  # 最小行高 10px
                    lines.append(gray_img[start:i, :])
                in_text = False

        if in_text and h - start > 10:
            lines.append(gray_img[start:h, :])

        # 如果没找到多行，返回原图
        return lines if len(lines) > 1 else [gray_img]

    def _ocr_single_line(self, gray_img):
        """识别单行文字"""
        h, w = gray_img.shape

        # resize 到 32 高度，保持宽高比
        target_h = 32
        target_w = max(32, int(w * target_h / h))
        # 限制最大宽度
        target_w = min(target_w, 2000)

        resized = cv2.resize(gray_img, (target_w, target_h))
        resized = resized.astype(np.float32) / 255.0

        # 转 tensor [1, 1, 32, W]
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.ocr_model(tensor)

        text = ctc_greedy_decode(output)[0]
        return text

    # ---------- Step 4: 行列组装 ----------

    def _assemble_table(self, cells_with_text):
        """
        将 cell + 文字组装为行列结构
        cells_with_text: [{'bbox': [x1,y1,x2,y2], 'text': '...', 'score': 0.9}, ...]

        返回: {
            'grid': [[cell, cell, ...], ...],  # 二维数组
            'n_rows': int,
            'n_cols': int,
            'cells': [...],  # 带行列信息的 cell 列表
        }
        """
        if not cells_with_text:
            return {'grid': [], 'n_rows': 0, 'n_cols': 0, 'cells': []}

        # 用 y1（上边界）聚类行号，用 x1（左边界）聚类列号
        # 这样跨行 cell 的 y1 会归到起始行
        y_tops = [c['bbox'][1] for c in cells_with_text]
        x_lefts = [c['bbox'][0] for c in cells_with_text]
        heights = [c['bbox'][3] - c['bbox'][1] for c in cells_with_text]
        widths = [c['bbox'][2] - c['bbox'][0] for c in cells_with_text]

        # 行高中位数、列宽中位数（用非跨行 cell 的高度）
        # 排除明显过高的 cell（> 中位数 * 1.5）来计算基准
        h_median_raw = np.median(heights) if heights else 1
        base_heights = [h for h in heights if h < h_median_raw * 1.5]
        base_widths = [w for w in widths if w < np.median(widths) * 1.5] if widths else [1]
        median_h = np.median(base_heights) if base_heights else h_median_raw
        median_w = np.median(base_widths) if base_widths else 1

        row_ids = self._cluster_values(y_tops, threshold=median_h * 0.5)
        col_ids = self._cluster_values(x_lefts, threshold=median_w * 0.3)

        n_rows = max(row_ids) + 1 if row_ids else 0
        n_cols = max(col_ids) + 1 if col_ids else 0

        # 给每个 cell 分配行列 + 计算 rowspan/colspan
        enriched_cells = []
        for i, cell in enumerate(cells_with_text):
            h = cell['bbox'][3] - cell['bbox'][1]
            w = cell['bbox'][2] - cell['bbox'][0]
            rowspan = max(1, round(h / median_h))
            colspan = max(1, round(w / median_w))

            enriched_cells.append({
                **cell,
                'row': row_ids[i],
                'col': col_ids[i],
                'rowspan': rowspan,
                'colspan': colspan,
            })

        # 构建网格
        grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in enriched_cells:
            r, c = cell['row'], cell['col']
            if r < n_rows and c < n_cols:
                if grid[r][c] is None:
                    grid[r][c] = cell

        return {
            'grid': grid,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'cells': enriched_cells,
        }

    def _cluster_values(self, values, threshold):
        """将数值聚类为整数编号"""
        if not values:
            return []
        sorted_vals = sorted(set(values))
        clusters = {}
        cluster_id = 0
        clusters[sorted_vals[0]] = cluster_id
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i - 1] > threshold:
                cluster_id += 1
            clusters[sorted_vals[i]] = cluster_id
        return [clusters[v] for v in values]

    # ---------- 端到端入口 ----------

    def extract(self, image_path):
        """
        端到端提取
        返回: {
            'image': str,
            'tables': [
                {
                    'bbox': [x1,y1,x2,y2],
                    'detection_score': float,
                    'n_rows': int,
                    'n_cols': int,
                    'grid': [[cell, ...], ...],
                    'cells': [{'bbox', 'text', 'score', 'row', 'col', 'rowspan', 'colspan'}, ...]
                }, ...
            ]
        }
        """
        image = Image.open(image_path).convert('RGB')
        print(f"  Processing: {os.path.basename(image_path)}")

        # Step 1: 检测表格
        tables = self._detect_tables(image)
        print(f"    Detected {len(tables)} table(s)")

        result = {'image': image_path, 'tables': []}

        for table_idx, table in enumerate(tables):
            x1, y1, x2, y2 = [int(v) for v in table['bbox']]
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)

            table_image = image.crop((x1, y1, x2, y2))

            # Step 2: 结构识别
            cells = self._recognize_structure(table_image)
            print(f"    Table {table_idx}: {len(cells)} cells")

            # Step 3: 逐 cell 识别文字
            cells_with_text = []
            for cell in cells:
                cx1, cy1, cx2, cy2 = [int(v) for v in cell['bbox']]
                # 裁剪 cell 图片
                cell_image = table_image.crop((cx1, cy1, cx2, cy2))
                text = self._recognize_text(cell_image)
                cells_with_text.append({
                    'bbox': cell['bbox'],
                    'score': cell['score'],
                    'text': text,
                })

            # Step 4: 组装行列
            assembled = self._assemble_table(cells_with_text)

            result['tables'].append({
                'bbox': table['bbox'],
                'detection_score': table['score'],
                'n_rows': assembled['n_rows'],
                'n_cols': assembled['n_cols'],
                'grid': assembled['grid'],
                'cells': assembled['cells'],
            })

        return result

    # ---------- 输出格式 ----------

    def to_html(self, result, output_path=None):
        """输出为 HTML 表格"""
        html_parts = ['<!DOCTYPE html><html><head>',
                      '<meta charset="utf-8">',
                      '<style>table{border-collapse:collapse;margin:20px}'
                      'td,th{border:1px solid #333;padding:6px 12px;text-align:left}'
                      'th{background:#f0f0f0}</style>',
                      '</head><body>']

        for table_idx, table in enumerate(result['tables']):
            html_parts.append(f'<h3>Table {table_idx + 1}</h3>')
            html_parts.append('<table>')

            # 标记被 rowspan/colspan 占用的格子
            occupied = set()
            for r in range(table['n_rows']):
                html_parts.append('<tr>')
                for c in range(table['n_cols']):
                    if (r, c) in occupied:
                        continue
                    cell = table['grid'][r][c] if r < len(table['grid']) and c < len(table['grid'][r]) else None
                    if cell:
                        rs = f' rowspan="{cell["rowspan"]}"' if cell['rowspan'] > 1 else ''
                        cs = f' colspan="{cell["colspan"]}"' if cell['colspan'] > 1 else ''
                        text = cell.get('text', '').replace('\n', '<br>')
                        html_parts.append(f'<td{rs}{cs}>{text}</td>')
                        # 标记占用
                        for dr in range(cell['rowspan']):
                            for dc in range(cell['colspan']):
                                if dr > 0 or dc > 0:
                                    occupied.add((r + dr, c + dc))
                    else:
                        html_parts.append('<td></td>')
                html_parts.append('</tr>')
            html_parts.append('</table>')

        html_parts.append('</body></html>')
        html = '\n'.join(html_parts)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"  HTML saved: {output_path}")
        return html

    def to_csv(self, result, output_path=None):
        """输出为 CSV"""
        import csv
        tables_csv = []
        for table_idx, table in enumerate(result['tables']):
            rows = []
            for r in range(table['n_rows']):
                row = []
                for c in range(table['n_cols']):
                    cell = table['grid'][r][c] if r < len(table['grid']) and c < len(table['grid'][r]) else None
                    row.append(cell.get('text', '') if cell else '')
                rows.append(row)
            tables_csv.append(rows)

        if output_path:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for table_idx, rows in enumerate(tables_csv):
                    if table_idx > 0:
                        writer.writerow([])
                    writer.writerow([f'--- Table {table_idx + 1} ---'])
                    for row in rows:
                        writer.writerow(row)
            print(f"  CSV saved: {output_path}")
        return tables_csv

    def to_json(self, result, output_path=None):
        """输出为 JSON"""
        # 序列化时去掉 grid（含 None，JSON 不友好），保留 cells
        json_result = {'image': result['image'], 'tables': []}
        for table in result['tables']:
            json_result['tables'].append({
                'bbox': table['bbox'],
                'detection_score': table['detection_score'],
                'n_rows': table['n_rows'],
                'n_cols': table['n_cols'],
                'cells': table['cells'],
            })

        json_str = json.dumps(json_result, indent=2, ensure_ascii=False)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"  JSON saved: {output_path}")
        return json_str


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='Table OCR Extraction Pipeline')
    parser.add_argument('--input', type=str, required=True, help='输入图片或目录')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--detector', type=str, default='runs/detection/best_model.pth')
    parser.add_argument('--structure', type=str, default='runs/structure/best_model.pth')
    parser.add_argument('--ocr', type=str, default=None, help='CRNN OCR 模型路径')
    args = parser.parse_args()

    # 默认 OCR 路径
    if args.ocr is None:
        # 尝试自动查找项目1的模型
        candidates = [
            '../项目1_简单OCR系统/runs/v2/best_model.pth',
            os.path.expanduser('~/autodl-tmp/项目1_简单OCR系统/runs/v2/best_model.pth'),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.ocr = c
                break
        if args.ocr is None:
            print("  [ERROR] 请指定 --ocr 路径（项目1的 CRNN 模型）")
            return

    print("\n  Initializing pipeline...")
    pipeline = TableOCRPipeline(args.detector, args.structure, args.ocr)

    os.makedirs(args.output, exist_ok=True)

    # 处理输入
    if os.path.isfile(args.input):
        images = [args.input]
    else:
        images = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]

    print(f"\n  Found {len(images)} image(s)\n")

    for img_path in images:
        try:
            result = pipeline.extract(img_path)
            base = os.path.splitext(os.path.basename(img_path))[0]

            pipeline.to_html(result, os.path.join(args.output, f'{base}.html'))
            pipeline.to_csv(result, os.path.join(args.output, f'{base}.csv'))
            pipeline.to_json(result, os.path.join(args.output, f'{base}.json'))

            # 打印预览
            for t_idx, table in enumerate(result['tables']):
                print(f"\n    Table {t_idx + 1} ({table['n_rows']}×{table['n_cols']}):")
                for r in range(min(table['n_rows'], 5)):
                    row_text = []
                    for c in range(table['n_cols']):
                        cell = table['grid'][r][c] if r < len(table['grid']) and c < len(table['grid'][r]) else None
                        text = cell.get('text', '')[:20] if cell else ''
                        row_text.append(text.ljust(20))
                    print(f"      | {'  |  '.join(row_text)}  |")
                if table['n_rows'] > 5:
                    print(f"      ... ({table['n_rows'] - 5} more rows)")

        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}")

    print(f"\n  Done! Results saved to: {args.output}/")


if __name__ == '__main__':
    main()
