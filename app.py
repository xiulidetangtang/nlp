# 完整的Flask应用 - 修复所有语法错误
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import traceback

# 导入你的模型文件 (根据实际路径调整)
try:
    from score import score, load_model
except ImportError:
    print("Warning: Could not import score functions. Please ensure score.py is in the correct path.")


    def score(text, model):
        # 临时占位符函数
        return [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


    def load_model(path):
        return None

app = Flask(__name__)
CORS(app, origins=["*"])  # 允许所有域名的跨域请求

# 全局变量存储模型
model = None


def initialize_model():
    """初始化模型"""
    global model
    try:
        print("Loading trained model...")
        model = load_model('./saved/model.pt')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


@app.route('/api/score_essay', methods=['POST'])
def score_essay_api():
    """
    API端点：为作文评分
    接收JSON格式的作文文本，返回六维度评分
    """
    try:
        # 添加调试信息
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        print(f"Raw data: {request.get_data()}")

        # 获取请求数据
        data = request.get_json(force=True)  # 添加force=True参数
        print(f"Parsed JSON: {data}")

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing essay text in request body',
                'success': False,
                'received_data': str(data)
            }), 400

        essay_text = data['text'].strip()
        print(f"Essay text length: {len(essay_text)}")

        if len(essay_text) < 10:  # 降低最小长度要求用于测试
            return jsonify({
                'error': 'Essay too short. Minimum 10 characters required.',
                'success': False
            }), 400

        # 检查模型是否加载
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the server.',
                'success': False
            }), 500

        # 调用你的评分函数
        print("Calling scoring model...")
        try:
            scores = score(essay_text, model)
            print(f"Model returned: {scores}")
            print(f"Scores type: {type(scores)}")
        except Exception as model_error:
            print(f"Error calling score function: {str(model_error)}")
            traceback.print_exc()
            return jsonify({
                'error': f'Error calling scoring model: {str(model_error)}',
                'success': False,
                'model_error': True
            }), 500

        # 确保scores是正确的格式
        score_dict = {}

        if isinstance(scores, list):
            # 如果返回列表，按顺序映射到维度
            dimension_names = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
            for i, name in enumerate(dimension_names):
                if i < len(scores):
                    score_dict[name] = float(scores[i])
                else:
                    score_dict[name] = 3.0  # 默认值

        elif isinstance(scores, dict):
            # 如果已经是字典格式
            score_dict = {k: float(v) for k, v in scores.items()}

        elif isinstance(scores, (int, float)):
            # 如果是单个数值
            single_score = float(scores)
            score_dict = {
                'cohesion': single_score,
                'syntax': single_score,
                'vocabulary': single_score,
                'phraseology': single_score,
                'grammar': single_score,
                'conventions': single_score
            }
        else:
            # 其他格式，使用默认值
            print(f"Unexpected scores format: {type(scores)}")
            score_dict = {
                'cohesion': 3.0,
                'syntax': 3.0,
                'vocabulary': 3.0,
                'phraseology': 3.0,
                'grammar': 3.0,
                'conventions': 3.0
            }

        # 计算总体分数
        overall_score = sum(score_dict.values()) / len(score_dict)

        result = {
            'success': True,
            'scores': score_dict,
            'overall_score': round(overall_score, 2),
            'word_count': len(essay_text.split()),
            'message': 'Essay scored successfully'
        }

        print(f"Returning result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error in score_essay_api: {str(e)}")
        traceback.print_exc()

        return jsonify({
            'error': f'Model error: {str(e)}',
            'success': False,
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate_feedback', methods=['POST'])
def generate_feedback_api():
    """
    API端点：生成详细反馈
    整合真实的大语言模型
    """
    try:
        data = request.get_json()
        essay_text = data['text']
        scores = data.get('scores', {})

        print(f"Generating feedback for essay length: {len(essay_text)}")
        print(f"Scores: {scores}")

        # 调用真实的大语言模型
        feedback = generate_real_ai_feedback(essay_text, scores)

        return jsonify({
            'success': True,
            'feedback': feedback
        })

    except Exception as e:
        print(f"Error in generate_feedback_api: {str(e)}")
        traceback.print_exc()

        return jsonify({
            'error': str(e),
            'success': False
        }), 500


def generate_real_ai_feedback(essay_text, scores):
    """
    使用真实的大语言模型生成反馈
    """

    # 方法1: 使用OpenAI GPT-4 (支持第三方API)
    try:
        import openai

        # 使用新的OpenAI客户端格式（支持第三方API）
        client = openai.OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://zzzzapi.com/v1"
        )

        # 构建详细的提示词
        prompt = f"""
你是一位专业的英语写作导师，正在为8-12年级的英语学习者提供写作反馈。请基于以下信息为学生提供详细、建设性的反馈：

学生作文：
{essay_text}

AI评分结果（满分5.0）：
- 连贯性 (Cohesion): {scores.get('cohesion', 'N/A'):.2f}
- 句法 (Syntax): {scores.get('syntax', 'N/A'):.2f}  
- 词汇 (Vocabulary): {scores.get('vocabulary', 'N/A'):.2f}
- 短语运用 (Phraseology): {scores.get('phraseology', 'N/A'):.2f}
- 语法 (Grammar): {scores.get('grammar', 'N/A'):.2f}
- 写作规范 (Conventions): {scores.get('conventions', 'N/A'):.2f}

请提供以下方面的反馈（用HTML格式，包含适当的标题和列表）：

1. **整体评估** - 总结文章的整体表现
2. **主要优点** - 识别学生做得好的地方
3. **需要改进的领域** - 基于评分指出具体问题
4. **具体建议** - 给出可操作的改进建议
5. **下一步学习重点** - 推荐学习方向

请确保反馈：
- 鼓励性且建设性
- 具体而非泛泛而谈
- 适合该年龄段学生的理解水平
- 包含具体的例子和建议
"""

        response = client.chat.completions.create(
            model="gpt-4",  # 或者根据你的API服务支持的模型调整
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    except ImportError:
        print("OpenAI library not installed. Run: pip install openai")
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        print("Make sure your API key and base_url are correct")

    # 方法2: 使用Claude API (Anthropic)
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key="YOUR_CLAUDE_API_KEY_HERE"  # 请替换为实际的API密钥
        )

        prompt = f"""请为这篇英语学习者的作文提供专业的写作反馈。

作文内容：
{essay_text}

AI评分（满分5.0）：
连贯性: {scores.get('cohesion', 0):.2f} | 句法: {scores.get('syntax', 0):.2f} | 词汇: {scores.get('vocabulary', 0):.2f}
短语: {scores.get('phraseology', 0):.2f} | 语法: {scores.get('grammar', 0):.2f} | 规范: {scores.get('conventions', 0):.2f}

请提供HTML格式的详细反馈，包括优点、改进建议和学习重点。"""

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    except ImportError:
        print("Anthropic library not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"Claude API error: {str(e)}")

    # 备用方案: 智能的基于规则的反馈
    print("Using intelligent rule-based feedback as fallback")
    return generate_intelligent_rule_based_feedback(essay_text, scores)


def generate_intelligent_rule_based_feedback(essay_text, scores):
    """
    智能的基于规则的反馈生成（备用方案）
    """
    word_count = len(essay_text.split())
    sentence_count = len([s for s in essay_text.split('.') if s.strip()])
    paragraph_count = len([p for p in essay_text.split('\n\n') if p.strip()])

    feedback_parts = []

    # 计算平均分
    avg_score = sum(scores.values()) / len(scores) if scores else 3.0

    # 整体评估
    feedback_parts.append('<h4 style="color: #4facfe; margin-bottom: 10px;">📊 Overall Assessment</h4>')
    if avg_score >= 4.0:
        assessment = f"Excellent work! Your essay demonstrates strong writing skills with an average score of {avg_score:.2f}/5.0."
    elif avg_score >= 3.5:
        assessment = f"Good job! Your essay shows solid writing ability with an average score of {avg_score:.2f}/5.0."
    elif avg_score >= 3.0:
        assessment = f"Your essay shows promise with an average score of {avg_score:.2f}/5.0, with several areas for improvement."
    else:
        assessment = f"Your essay has potential with an average score of {avg_score:.2f}/5.0. Focus on the improvement areas below."

    feedback_parts.append(
        f'<p>{assessment} The essay contains {word_count} words across {paragraph_count} paragraphs.</p>')

    # 主要优点
    feedback_parts.append('<h4 style="color: #28a745; margin: 20px 0 10px 0;">✅ Strengths</h4>')
    feedback_parts.append('<ul style="margin-left: 20px;">')

    # 基于分数识别优点
    if scores:
        best_aspect = max(scores.keys(), key=lambda k: scores[k])
        feedback_parts.append(
            f'<li>Your strongest area is <strong>{best_aspect}</strong> (score: {scores[best_aspect]:.2f})</li>')

    if word_count > 200:
        feedback_parts.append('<li>Good essay length showing developed ideas</li>')
    if paragraph_count >= 4:
        feedback_parts.append('<li>Well-structured with multiple paragraphs</li>')
    if any(word in essay_text.lower() for word in ['however', 'furthermore', 'therefore', 'consequently']):
        feedback_parts.append('<li>Good use of transition words</li>')

    feedback_parts.append('</ul>')

    # 需要改进的领域
    feedback_parts.append('<h4 style="color: #ffc107; margin: 20px 0 10px 0;">🎯 Areas for Improvement</h4>')
    feedback_parts.append('<ul style="margin-left: 20px;">')

    # 基于最低分数提供建议
    if scores:
        weakest_aspect = min(scores.keys(), key=lambda k: scores[k])
        feedback_parts.append(
            f'<li>Focus on improving <strong>{weakest_aspect}</strong> (current score: {scores[weakest_aspect]:.2f})</li>')

    if scores.get('cohesion', 3) < 3.0:
        feedback_parts.append('<li>Work on connecting ideas between paragraphs more clearly</li>')
    if scores.get('vocabulary', 3) < 3.0:
        feedback_parts.append('<li>Try using more varied and sophisticated vocabulary</li>')
    if scores.get('grammar', 3) < 3.0:
        feedback_parts.append('<li>Review grammar rules and proofread more carefully</li>')

    feedback_parts.append('</ul>')

    # 具体建议
    feedback_parts.append('<h4 style="color: #dc3545; margin: 20px 0 10px 0;">💡 Specific Recommendations</h4>')
    feedback_parts.append('<ul style="margin-left: 20px;">')
    feedback_parts.append('<li>Read your essay aloud to catch errors and improve flow</li>')
    feedback_parts.append('<li>Use a variety of sentence structures (simple, compound, complex)</li>')
    feedback_parts.append('<li>Support your main points with specific examples</li>')
    feedback_parts.append('<li>Check spelling and punctuation before submitting</li>')
    feedback_parts.append('</ul>')

    # 下一步学习重点
    feedback_parts.append('<h4 style="color: #6f42c1; margin: 20px 0 10px 0;">📚 Next Steps</h4>')
    if avg_score < 3.0:
        next_steps = "Focus on basic writing fundamentals: clear topic sentences, supporting details, and proper grammar."
    elif avg_score < 3.5:
        next_steps = "Work on developing ideas more fully and using more sophisticated vocabulary and sentence structures."
    else:
        next_steps = "Refine your style by varying sentence patterns and incorporating more advanced transitional phrases."

    feedback_parts.append(f'<p><strong>Priority:</strong> {next_steps}</p>')
    feedback_parts.append(
        '<p><strong>Practice:</strong> Write one paragraph daily focusing on your weakest area, and read essays by skilled writers in your areas of interest.</p>')

    return ''.join(feedback_parts)


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'endpoints': ['/api/score_essay', '/api/generate_feedback', '/health']
    })


@app.route('/', methods=['GET'])
def home():
    """主页"""
    return jsonify({
        'message': 'English Language Learner Essay Scoring API',
        'version': '1.0',
        'endpoints': {
            'score_essay': '/api/score_essay (POST)',
            'generate_feedback': '/api/generate_feedback (POST)',
            'health_check': '/health (GET)'
        }
    })


if __name__ == '__main__':
    # 启动时初始化模型
    model_loaded = initialize_model()
    if not model_loaded:
        print("Warning: Model could not be loaded. Some features may not work correctly.")

    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
