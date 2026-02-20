#!/usr/bin/env python3
"""
智能相册整理工具 - AI驱动的照片管理系统
结合传统CV算法 + Qwen 2.5 7B 本地大模型
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import requests

class PhotoAnalyzer:
    """照片分析器:提取图像特征"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        
    def calculate_perceptual_hash(self, image_path: str, hash_size: int = 8) -> str:
        """计算感知哈希(用于相似图片检测)"""
        try:
            img = Image.open(image_path).convert('L')  # 转灰度
            img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img).flatten()
            avg = pixels.mean()
            hash_binary = ''.join(['1' if p > avg else '0' for p in pixels])
            return hash_binary
        except Exception as e:
            print(f"❌ 无法处理 {image_path}: {e}")
            return ""
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算汉明距离(衡量相似度)"""
        if len(hash1) != len(hash2):
            return -1
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def calculate_sharpness(self, image_path: str) -> float:
        """计算图像清晰度(拉普拉斯方差)"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            return float(laplacian.var())
        except:
            return 0.0
    
    def extract_features(self, image_path: str) -> Dict:
        """提取图像特征"""
        try:
            # 基础信息
            stat = os.stat(image_path)
            img = Image.open(image_path)
            
            # 提取EXIF信息
            exif = img._getexif() or {}
            date_taken = None
            if exif:
                date_str = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
                if date_str:
                    try:
                        date_taken = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except:
                        pass
            
            features = {
                'path': image_path,
                'filename': os.path.basename(image_path),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'width': img.width,
                'height': img.height,
                'resolution': f"{img.width}x{img.height}",
                'format': img.format,
                'mode': img.mode,
                'date_taken': date_taken.isoformat() if date_taken else None,
                'file_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'sharpness': round(self.calculate_sharpness(image_path), 2),
                'perceptual_hash': self.calculate_perceptual_hash(image_path)
            }
            
            # 判断是否为截图
            filename_lower = os.path.basename(image_path).lower()
            features['is_screenshot'] = any(keyword in filename_lower 
                                           for keyword in ['screenshot', '截图', 'screen shot', 'img_'])
            
            return features
        except Exception as e:
            print(f"❌ 特征提取失败 {image_path}: {e}")
            return None


class QwenClient:
    """Qwen 2.5 7B 本地模型调用客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5:7b"
        
    def analyze_photo_group(self, photos: List[Dict]) -> Dict:
        """让Qwen分析一组相似照片,给出保留建议"""
        
        # 构建Prompt
        photo_descriptions = []
        for i, photo in enumerate(photos, 1):
            desc = f"""照片{i}:
- 文件名: {photo['filename']}
- 分辨率: {photo['resolution']}
- 文件大小: {photo['size_mb']} MB
- 清晰度分数: {photo['sharpness']}
- 拍摄时间: {photo['date_taken'] or '未知'}
"""
            photo_descriptions.append(desc)
        
        prompt = f"""你是一个专业的照片管理助手。以下是一组相似的照片(可能是连拍或重复):

{chr(10).join(photo_descriptions)}

请分析这些照片并给出建议:
1. 推荐保留哪张(或哪几张)?为什么?
2. 其他照片可以删除的理由?

请用JSON格式回复:
{{
  "recommended_keep": [照片编号列表],
  "recommended_delete": [照片编号列表],
  "reason": "简洁的推荐理由(一句话)"
}}

只返回JSON,不要其他内容。"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # 尝试解析JSON
                try:
                    # 提取JSON部分
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        decision = json.loads(json_str)
                        return decision
                except:
                    pass
                
                # 如果JSON解析失败,返回原始响应
                return {
                    "recommended_keep": [1],
                    "recommended_delete": list(range(2, len(photos) + 1)),
                    "reason": "AI分析建议保留第一张(基于清晰度和质量)",
                    "raw_response": response_text
                }
            else:
                print(f"❌ Qwen API错误: {response.status_code}")
                return self._fallback_decision(photos)
                
        except Exception as e:
            print(f"❌ Qwen调用失败: {e}")
            return self._fallback_decision(photos)
    
    def _fallback_decision(self, photos: List[Dict]) -> Dict:
        """备用决策:基于清晰度选择"""
        sorted_photos = sorted(enumerate(photos, 1), 
                              key=lambda x: x[1]['sharpness'], 
                              reverse=True)
        keep = [sorted_photos[0][0]]
        delete = [p[0] for p in sorted_photos[1:]]
        return {
            "recommended_keep": keep,
            "recommended_delete": delete,
            "reason": "基于清晰度自动选择(Qwen未响应)"
        }
    
    def analyze_screenshot(self, photo: Dict) -> Dict:
        """判断截图是否为临时文件"""
        prompt = f"""分析以下截图信息,判断是否为临时功能性文件(如验证码、订单、快递信息等):

文件名: {photo['filename']}
文件大小: {photo['size_mb']} MB
分辨率: {photo['resolution']}

请用JSON格式回复:
{{
  "is_temporary": true/false,
  "category": "验证码/订单/快递/其他",
  "suggested_action": "建议保留7天后删除 / 可以立即删除 / 建议长期保留",
  "confidence": 0-100的置信度
}}

只返回JSON。"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            
            return {
                "is_temporary": True,
                "category": "未知",
                "suggested_action": "建议保留30天后删除",
                "confidence": 50
            }
        except:
            return {
                "is_temporary": True,
                "category": "未知",
                "suggested_action": "建议保留30天后删除",
                "confidence": 50
            }


class PhotoOrganizer:
    """照片整理主控制器"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.analyzer = PhotoAnalyzer()
        self.qwen = QwenClient()
        self.photos = []
        self.similar_groups = []
        self.screenshots = []
        
    def scan_photos(self):
        """扫描文件夹中的所有照片"""
        print(f"📂 正在扫描文件夹: {self.folder_path}")
        
        for file_path in self.folder_path.rglob('*'):
            if file_path.suffix.lower() in self.analyzer.supported_formats:
                features = self.analyzer.extract_features(str(file_path))
                if features:
                    self.photos.append(features)
                    if features['is_screenshot']:
                        self.screenshots.append(features)
        
        print(f"✅ 找到 {len(self.photos)} 张照片 (其中 {len(self.screenshots)} 张截图)")
    
    def find_similar_groups(self, threshold: int = 10):
        """找出相似照片组"""
        print(f"🔍 正在检测相似照片...")
        
        processed = set()
        
        for i, photo1 in enumerate(self.photos):
            if i in processed or not photo1['perceptual_hash']:
                continue
            
            group = [photo1]
            group_indices = [i]
            
            for j, photo2 in enumerate(self.photos[i+1:], i+1):
                if j in processed or not photo2['perceptual_hash']:
                    continue
                
                distance = self.analyzer.hamming_distance(
                    photo1['perceptual_hash'], 
                    photo2['perceptual_hash']
                )
                
                if distance >= 0 and distance <= threshold:
                    group.append(photo2)
                    group_indices.append(j)
            
            if len(group) > 1:
                self.similar_groups.append(group)
                processed.update(group_indices)
        
        print(f"✅ 找到 {len(self.similar_groups)} 组相似照片")
    
    def analyze_with_ai(self):
        """使用Qwen分析所有照片"""
        print(f"🤖 正在调用 Qwen 2.5 进行智能分析...")
        
        # 分析相似照片组
        for i, group in enumerate(self.similar_groups, 1):
            print(f"  分析第 {i}/{len(self.similar_groups)} 组...")
            decision = self.qwen.analyze_photo_group(group)
            
            # 将决策结果添加到照片信息中
            for photo in group:
                photo['group_id'] = i
                photo['ai_decision'] = decision
        
        # 分析截图
        if self.screenshots:
            print(f"  分析 {len(self.screenshots)} 张截图...")
            for screenshot in self.screenshots[:10]:  # 限制数量避免时间过长
                screenshot['screenshot_analysis'] = self.qwen.analyze_screenshot(screenshot)
    
    def generate_report(self, output_path: str = "photo_report.html"):
        """生成HTML可视化报告"""
        print(f"📊 正在生成报告...")
        
        # 计算统计数据
        total_photos = len(self.photos)
        total_size_mb = sum(p['size_mb'] for p in self.photos)
        
        deletable_photos = []
        for group in self.similar_groups:
            decision = group[0].get('ai_decision', {})
            delete_indices = decision.get('recommended_delete', [])
            for idx in delete_indices:
                if 0 < idx <= len(group):
                    deletable_photos.append(group[idx-1])
        
        potential_savings_mb = sum(p['size_mb'] for p in deletable_photos)
        
        # 生成HTML
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能相册整理报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 1.1em;
        }}
        .section {{
            padding: 40px;
        }}
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 30px;
            color: #333;
            border-left: 5px solid #667eea;
            padding-left: 20px;
        }}
        .photo-group {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .photo-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .photo-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .photo-card.keep {{
            border: 3px solid #10b981;
        }}
        .photo-card.delete {{
            border: 3px solid #ef4444;
            opacity: 0.7;
        }}
        .photo-info {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .ai-reason {{
            background: #e0e7ff;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-style: italic;
            color: #4c51bf;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 10px;
        }}
        .badge-keep {{ background: #d1fae5; color: #065f46; }}
        .badge-delete {{ background: #fee2e2; color: #991b1b; }}
        .screenshot-section {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📸 智能相册整理报告</h1>
            <p>AI驱动的照片管理分析 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_photos}</div>
                <div class="stat-label">扫描照片总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.similar_groups)}</div>
                <div class="stat-label">相似照片组</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(deletable_photos)}</div>
                <div class="stat-label">建议删除</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{round(potential_savings_mb, 1)}MB</div>
                <div class="stat-label">可释放空间</div>
            </div>
        </div>
"""
        
        # 相似照片组
        if self.similar_groups:
            html += """
        <div class="section">
            <h2 class="section-title">🔍 相似照片检测结果</h2>
"""
            for i, group in enumerate(self.similar_groups, 1):
                decision = group[0].get('ai_decision', {})
                keep_indices = decision.get('recommended_keep', [1])
                reason = decision.get('reason', '未提供理由')
                
                html += f"""
            <div class="photo-group">
                <h3>📁 照片组 #{i} ({len(group)} 张)</h3>
                <div class="ai-reason">🤖 AI建议: {reason}</div>
                <div class="photo-grid">
"""
                for j, photo in enumerate(group, 1):
                    badge_class = "keep" if j in keep_indices else "delete"
                    badge_text = "✅ 保留" if j in keep_indices else "❌ 删除"
                    
                    html += f"""
                    <div class="photo-card {badge_class}">
                        <div class="photo-info">
                            <strong>{photo['filename']}</strong><br>
                            分辨率: {photo['resolution']}<br>
                            大小: {photo['size_mb']} MB<br>
                            清晰度: {photo['sharpness']}
                        </div>
                        <span class="badge badge-{badge_class}">{badge_text}</span>
                    </div>
"""
                html += """
                </div>
            </div>
"""
            html += "</div>"
        
        # 截图分析
        if self.screenshots:
            html += """
        <div class="section screenshot-section">
            <h2 class="section-title">📱 截图文件分析</h2>
            <p style="margin-bottom: 20px;">检测到 <strong>{}</strong> 张截图文件</p>
""".format(len(self.screenshots))
            
            for screenshot in self.screenshots[:10]:
                analysis = screenshot.get('screenshot_analysis', {})
                action = analysis.get('suggested_action', '建议保留30天后删除')
                
                html += f"""
            <div class="photo-card" style="margin-bottom: 15px;">
                <div class="photo-info">
                    <strong>{screenshot['filename']}</strong><br>
                    建议操作: {action}
                </div>
            </div>
"""
            html += "</div>"
        
        html += """
    </div>
</body>
</html>
"""
        
        output_file = Path(output_path)
        output_file.write_text(html, encoding='utf-8')
        print(f"✅ 报告已生成: {output_file.absolute()}")
        
        return str(output_file.absolute())


def main():
    """主函数"""
    print("=" * 60)
    print("🎨 智能相册整理工具 v1.0")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("❌ 用法: python photo_organizer.py <照片文件夹路径>")
        print("📝 示例: python photo_organizer.py ~/Pictures/test_photos")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        sys.exit(1)
    
    # 创建整理器
    organizer = PhotoOrganizer(folder_path)
    
    # 执行分析流程
    organizer.scan_photos()
    organizer.find_similar_groups()
    organizer.analyze_with_ai()
    report_path = organizer.generate_report()
    
    print()
    print("=" * 60)
    print(f"🎉 分析完成! 请打开报告查看结果:")
    print(f"📄 {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
