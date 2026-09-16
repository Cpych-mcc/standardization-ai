import streamlit as st
import pandas as pd
import os
from openai import OpenAI  # 这里使用 OpenAI 格式的 SDK，兼容大多数国产大模型

# --- 页面配置 ---
st.set_page_config(page_title="标准化工作智能助手", layout="wide", page_icon="⚖️")

# --- 侧边栏：配置与说明 ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key = st.text_input("请输入大模型 API Key", type="password", help="请输入兼容 OpenAI 格式的 API Key")
    base_url = st.text_input("API Base URL", value="https://api.deepseek.com") # 默认 DeepSeek，可改为通义千问等
    model_name = st.text_input("模型名称", value="deepseek-chat")
    
    st.markdown("---")
    st.info("💡 **使用指南**：\n1. 左侧选择任务场景\n2. 填入原始材料\n3. 点击生成获取专业文档")

# --- 核心 Prompt 库 (直接提取自你的 Excel) ---
PROMPT_TEMPLATES = {
    "国标草案研讨会会议纪要": """
你是一名资深的标准化工程师助理，负责将会议原始材料精准转化为会议纪要。你的核心职责是“忠实整理”，而非“创作”。

【任务】
根据我提供的原始材料，严格按模板生成纪要。

【绝对禁令（防幻觉核心）】
1. **零容忍虚构**：你生成的所有会议摘要、具体内容、后续安排、附录意见，必须能在原始材料中找到直接对应的字眼或事实依据。**严禁利用你自身的训练知识去补充“标准草案通常包含的内容”**。
2. **拒绝填坑**：如果原始材料中缺少某个章节（比如没有讨论后续工作安排），严禁编造，直接在该部分写“材料未涉及，待现场确认”。
3. **拒绝通用套话**：不要写“会议围绕XX进行了深入讨论”这种没有信息增量的废话，只写材料里明确提到的具体修订动作或结论。

【输出格式】
严格按以下结构输出，栏目名称不得改动：

一、会议时间
（直接填写，若材料未提及则写“待确认”）

二、会议形式
（直接填写，若材料未提及则写“待确认”）

三、会议内容

（一）会议摘要
（从材料中提炼核心议题和共识，150字内。若无足够信息，直接写“材料未涉及”）

（二）具体内容
【按“1、议题大类 -> （1）子议题 -> 具体结论”层级输出。直接陈述修改决定，严禁出现“某某提出/认为”。句式参考：将……修订为……；删除……；合并……。】
【注意：如果材料里只提到了某个议题，但没有具体的修改结论，就写“该议题讨论了……，具体结论待确认”。】

（三）后续工作安排
不用表格，直接分点列出。
每一个分点必须按：动作概括 + 承接主体（机构名或角色名，严禁具体人名）+ 时间节点 + 材料提交要求 输出。
【注意：如果材料里没有明确的部署，直接写“材料中未涉及后续工作安排”。】

四、参会人员名单及意见汇总
（直接列名单，并汇总各方核心意见。此处对应会议照片位置，留标记【此处插入会议照片】）

五、附录
1. 会议研讨意见采纳情况表
（提取、去重、合并输入中的“多方反馈意见”，按六列表格输出：序号、标准章节、修改意见、提出单位、提出人员、采纳情况）
【注意：如果输入中根本没有批注意见表，直接写“未提供意见汇总材料，无法生成此表”。】
2. 参会人员名单

【约束条件】
- 涉及标准条款、术语、编号时，保持原文准确，不做同义替换。
- 无法确认的信息一律标注“待确认”。
- **原始记录里没有的，绝不允许凭空捏造。**
""",

    "草案批注意见提取与采纳情况比对": """
你是一名标准草案意见处理专员，擅长处理多模态输入，能结合批注截图描述、原草案文本和修改后草案文本，精准提取所有批注意见，并根据版本比对得出采纳情况。

【任务】
根据我提供的来源单位、原草案文本、修改后草案文本及批注截图描述，提取所有批注意见，统一转写为标准书面语，并对比推断出是否采纳，按指定表格输出。

【处理规则（核心）】
1. 看图识意见：仅提取描述中明确标记为批注、建议、修改意见的内容。忽略描述中的草案正文。
2. 对号入座：每条意见务必结合原草案文本，识别其针对的条款号（如“3.1”、“表2”等）。如果是总体意见，章节写“总体”。
3. 对比定采纳：将提取出的修改意见，与“修改后草案文本”进行比对。
   - 若修改稿体现了该建议，采纳情况写“已采纳”，并在比对说明中简述如何修改的。
   - 若修改稿部分体现，写“部分采纳”。
   - 若修改稿未体现，写“未采纳”（或“待确认”，如果无法判定的话）。
4. 书面化改造：将批注中的口语化表达（如“这儿不对”、“换掉”）改写为标准化书面语（如“建议修改本条款表述”）。

【输出格式】
严格按以下表格输出，不要添加额外解释：

| 序号 | 标准章节 | 修改意见 | 提出单位 | 提出人员 | 采纳情况 | 比对说明/备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

【约束条件】
- 严禁把原草案正文当作批注意见输出。
- 严禁凭空捏造提出单位、条款号或修改建议。不确定的写“待确认”。
- 提取出的意见必须保持原意，不得擅自增加或删减。
"""
}

# --- 主界面逻辑 ---
st.title("⚖️ 标准化工作智能助手")
st.markdown("基于 Vibe Coding 构建的自动化工作流，集成会议纪要整理与草案意见比对功能。")

# 1. 选择场景
scene = st.selectbox("请选择工作场景", options=list(PROMPT_TEMPLATES.keys()))

# 2. 动态展示输入框
st.subheader("📝 原始材料输入")

if scene == "国标草案研讨会会议纪要":
    meeting_name = st.text_input("会议名称")
    meeting_time = st.text_input("会议时间")
    attendees = st.text_area("参会人员及角色")
    raw_record = st.text_area("原始记录（转写稿/速记/草稿）", height=300, placeholder="请粘贴会议录音转写文字或速记内容...")
    feedback = st.text_area("多方反馈意见（可选）", height=150, placeholder="请粘贴批注或意见表内容...")
    
    input_content = f"会议名称：{meeting_name}\n时间：{meeting_time}\n参会人员：{attendees}\n\n原始记录：\n{raw_record}\n\n反馈意见：\n{feedback}"

elif scene == "草案批注意见提取与采纳情况比对":
    source_unit = st.text_input("来源单位/专家")
    original_text = st.text_area("原草案文本（相关段落）", height=150)
    modified_text = st.text_area("修改后草案文本（相关段落）", height=150)
    screenshot_desc = st.text_area("批注截图内容描述（OCR识别结果或人工描述）", height=150, placeholder="例如：图片中在第3.1条旁边有批注‘这里表述不清，建议修改’...")
    
    input_content = f"来源单位：{source_unit}\n原草案：\n{original_text}\n\n修改稿：\n{modified_text}\n\n截图描述：\n{screenshot_desc}"

# 3. 生成按钮
if st.button("✨ 开始生成", type="primary"):
    if not api_key:
        st.error("请先在左侧侧边栏输入 API Key")
    elif not input_content.strip():
        st.warning("请输入原始材料内容")
    else:
        with st.spinner("AI 正在思考并撰写文档..."):
            try:
                # 初始化客户端
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                # 获取对应 Prompt
                system_prompt = PROMPT_TEMPLATES[scene]
                
                # 调用大模型
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_content}
                    ],
                    temperature=0.1 # 低温度，确保严谨性
                )
                
                result = response.choices[0].message.content
                
                # 展示结果
                st.success("生成完成！")
                st.markdown("### 📄 生成结果预览")
                st.markdown(result)
                
                # 下载按钮
                st.download_button(
                    label="📥 下载为 Markdown 文件",
                    data=result,
                    file_name=f"{scene}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"调用出错：{e}")

# --- 底部 ---
st.markdown("---")
st.caption("Vibe Coding 实践项目 | 标准化信息监控与处理中枢 v1.0")