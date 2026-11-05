"""
Demo 01-02 · 消息与提示词模板
============================
知识点
  1. 四种核心消息 System / Human / AI / Tool —— 消息列表是 LLM 应用的“通用数据结构”
  2. 多模态：用标准 content_blocks 构造图文混合消息（与厂商无关）
  3. ChatPromptTemplate：变量插值 + MessagesPlaceholder 注入历史；模板本身就是 Runnable
  4. partial 预填变量、few-shot 示例模板
  5. trim_messages：按 token 预算裁剪历史（上下文工程最基础的操作）
对应文档：docs/03-详设篇/01-模型-消息-提示词-LCEL.md
运行：uv run demos/01_core/02_messages_and_prompts.py
"""

from datetime import date

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show, show_messages


def demo_message_types() -> None:
    section("1. 四种核心消息")
    history = [
        SystemMessage("你是天气助手"),  # 系统指令：角色、规则、输出格式
        HumanMessage("北京天气如何？"),  # 用户输入
        AIMessage(  # 模型输出：可以带 tool_calls（模型“想”调用的工具）
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}],
        ),
        ToolMessage("晴，22℃", tool_call_id="call_1", name="get_weather"),  # 工具结果，必须用 tool_call_id 回指
        AIMessage("北京今天晴，22℃。"),
    ]
    show_messages(history)
    show("消息 type 字段", [m.type for m in history])

    section("2. 多模态消息：标准 content_blocks（同一写法适配各家模型）")
    multimodal = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "描述这张图片"},
            {"type": "image", "url": "https://example.com/cat.png"},
            # 也可以用 base64：{"type": "image", "base64": "...", "mime_type": "image/png"}
        ]
    )
    show("多模态 content_blocks", multimodal.content_blocks)


def demo_chat_prompt_template() -> None:
    section("3. ChatPromptTemplate + MessagesPlaceholder")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是{company}的客服，今天是{today}。回答不超过 50 字。"),
            MessagesPlaceholder("history"),  # 运行时把历史消息整段插进来
            ("human", "{question}"),
        ]
    )
    show("模板需要的变量", sorted(prompt.input_variables))

    # partial：预先固定一部分变量（可以是值，也可以是函数，每次格式化时求值）
    prompt = prompt.partial(company="星河科技", today=lambda: date.today().isoformat())
    history = [HumanMessage("你们周末上班吗？"), AIMessage("周末休息，工作日 9:00-18:00 在线。")]
    messages = prompt.format_messages(history=history, question="那我周六提的工单什么时候处理？")
    show_messages(messages)

    # 模板是 Runnable，可以直接用 | 接模型
    chain = prompt | get_chat_model()
    reply = chain.invoke({"history": history, "question": "那我周六提的工单什么时候处理？"})
    show("模型回答", reply.text)


def demo_few_shot() -> None:
    section("4. Few-shot：用示例“教”模型输出格式")
    examples = [
        {"input": "这家店的菜太难吃了", "output": "负面|口味"},
        {"input": "快递第二天就到了，点赞", "output": "正面|物流"},
        {"input": "客服态度很好，耐心解答", "output": "正面|服务"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    few_shot = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "对用户评论做情感与维度分类，只输出“情感|维度”，不要解释。"),
            few_shot,  # 展开后就是 3 轮 human/ai 示例对话
            ("human", "{input}"),
        ]
    )
    chain = prompt | get_chat_model()
    for text in ["包装破损，里面的杯子碎了", "价格比别家便宜一半"]:
        show(text, chain.invoke({"input": text}).text.strip())


def demo_trim_messages() -> None:
    section("5. trim_messages：按 token 预算裁剪历史")
    history = [SystemMessage("你是一个记忆力很好的助手")]
    for i in range(1, 11):
        history += [HumanMessage(f"第 {i} 轮问题：请记住数字 {i * 7}"), AIMessage(f"好的，记住了 {i * 7}")]

    trimmed = trim_messages(
        history,
        max_tokens=120,  # token 预算
        token_counter=count_tokens_approximately,  # 近似计数，零开销；也可以传模型本身做精确计数
        strategy="last",  # 保留最近的消息
        include_system=True,  # 无论如何保留 system 消息
        start_on="human",  # 裁剪后第一条非 system 消息必须是 human，保证对话结构合法
    )
    show("裁剪前消息数 / 近似 token", f"{len(history)} / {count_tokens_approximately(history)}")
    show("裁剪后消息数 / 近似 token", f"{len(trimmed)} / {count_tokens_approximately(trimmed)}")
    show_messages(trimmed[:3])


def main() -> None:
    banner("Demo 01-02 · 消息与提示词模板", "Messages / ChatPromptTemplate / few-shot / trim_messages")
    demo_message_types()
    demo_chat_prompt_template()
    demo_few_shot()
    demo_trim_messages()


if __name__ == "__main__":
    main()
