"""
Demo 01-04 · 结构化输出（Structured Output）
==========================================
知识点
  1. with_structured_output：Pydantic / TypedDict / JSON Schema 三种 schema 写法
  2. include_raw=True：同时拿到原始消息、解析结果、解析错误（便于降级处理）
  3. 实现原理：method="function_calling" 本质是“强制调用一个以 schema 为参数的工具”；
     method="json_schema" 则走厂商原生结构化输出（需要厂商支持）
  4. 对比老方法：PydanticOutputParser + 格式说明写进 prompt（不依赖工具调用，但更脆弱）
对应文档：docs/03-详设篇/02-工具与结构化输出.md
运行：uv run demos/01_core/04_structured_output.py
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()

RESUME = """张伟，32 岁，现居深圳。2016 年毕业于华中科技大学计算机专业。
先后在腾讯做后端开发 4 年、在一家 AI 创业公司做大模型应用架构师 3 年，
熟悉 Java、Python、LangChain、LangGraph、Milvus，带过 8 人团队。期望薪资 35K。"""


# ---- 写法 1：Pydantic（推荐）：有类型校验、字段描述会进入 schema 帮助模型理解 ----
class Experience(BaseModel):
    company: str = Field(description="公司名称")
    role: str = Field(description="职位")
    years: int = Field(description="工作年限")


class Candidate(BaseModel):
    """候选人简历信息"""

    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    city: str = Field(description="现居城市")
    skills: list[str] = Field(description="技术栈关键词")
    experiences: list[Experience] = Field(description="工作经历，按时间顺序")
    level: Literal["初级", "中级", "高级", "专家"] = Field(description="综合判断的职级")


# ---- 写法 2：TypedDict：返回普通 dict，Annotated 第二个参数是默认值，第三个是描述 ----
class Sentiment(TypedDict):
    """评论情感分析结果"""

    label: Annotated[Literal["正面", "负面", "中性"], ..., "情感倾向"]
    score: Annotated[float, ..., "置信度 0~1"]
    keywords: Annotated[list[str], ..., "支撑判断的关键词"]


# ---- 写法 3：原生 JSON Schema dict：适合 schema 来自配置中心/前端的场景 ----
TICKET_SCHEMA = {
    "title": "Ticket",
    "description": "从用户反馈中提取工单",
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["bug", "feature", "question"]},
        "priority": {"type": "integer", "description": "1 最高，5 最低"},
        "summary": {"type": "string", "description": "20 字以内的摘要"},
    },
    "required": ["category", "priority", "summary"],
}


def demo_three_schemas() -> None:
    section("1. Pydantic：返回模型实例（嵌套对象、枚举都支持）")
    candidate = llm.with_structured_output(Candidate).invoke(f"从简历中提取信息：\n{RESUME}")
    show("类型", type(candidate).__name__)
    show("结果", candidate.model_dump())

    section("2. TypedDict：返回 dict")
    sentiment = llm.with_structured_output(Sentiment).invoke("分析评论：物流很快，但是耳机音质一般，有点失望")
    show("结果", sentiment)

    section("3. JSON Schema：返回 dict")
    ticket = llm.with_structured_output(TICKET_SCHEMA).invoke("APP 一点支付就闪退，已经三天了，急！")
    show("结果", ticket)


def demo_include_raw() -> None:
    section("4. include_raw=True：原始消息 + 解析结果 + 解析错误")
    out = llm.with_structured_output(Sentiment, include_raw=True).invoke("这个产品还行吧，不好不坏")
    show("返回的 key", list(out))
    show("parsed", out["parsed"])
    show("parsing_error", out["parsing_error"])
    # 揭秘：raw 就是一条带 tool_calls 的 AIMessage —— 结构化输出本质是“强制工具调用”
    show("raw.tool_calls[0].name", out["raw"].tool_calls[0]["name"] if out["raw"].tool_calls else None)


def demo_output_parser() -> None:
    section("5. 老方法：PydanticOutputParser + 格式说明（不依赖工具调用能力）")
    parser = PydanticOutputParser(pydantic_object=Experience)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "从文本中提取信息。\n{format_instructions}"), ("human", "{text}")]
    ).partial(format_instructions=parser.get_format_instructions())
    show("自动生成的格式说明（节选）", parser.get_format_instructions()[:160] + "…")
    exp = (prompt | llm | parser).invoke({"text": "我在字节跳动当了 3 年的算法工程师"})
    show("解析结果", exp.model_dump())
    print("对比：with_structured_output 走工具调用/原生约束，成功率和可维护性都更好，新项目优先用它。")


def main() -> None:
    banner("Demo 01-04 · 结构化输出", "with_structured_output / include_raw / OutputParser")
    demo_three_schemas()
    demo_include_raw()
    demo_output_parser()


if __name__ == "__main__":
    main()
