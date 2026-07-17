import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool, ScrapeWebsiteTool

# 1. 加载环境变量
load_dotenv()

# 2. 初始化大模型大脑
custom_llm = LLM(
    model=f"openai/{os.getenv('CUSTOM_MODEL_NAME')}", 
    api_key=os.getenv("CUSTOM_API_KEY"),
    base_url=os.getenv("CUSTOM_BASE_URL"),
    temperature=0.3
)

# 3. 初始化工具
search_tool = TavilySearchTool()
scrape_tool = ScrapeWebsiteTool()

# ==================== 角色 1：情报搜集专家 ====================
data_scout = Agent(
    role='网络情报搜集专家',
    goal='在知乎、Reddit、V2EX 等各大社区搜集关于【{topic}】的最新真实讨论',
    backstory='你是一个敏锐的情报收集者。你擅长使用高级搜索指令，在嘈杂的网络社区中，定位到用户吐槽最真实、最具体的帖子，并爬取其详细内容。',
    verbose=True,
    allow_delegation=False,
    llm=custom_llm,
    tools=[search_tool, scrape_tool]
)

# ==================== 角色 2：资深产品分析师 ====================
product_analyst = Agent(
    role='资深 AI 产品运营专家',
    goal='将搜集到的社区原始舆情过滤脱水，提取核心痛点并输出产品迭代机会',
    backstory='你曾在知名大厂担任资深产品运营。你有一双火眼金睛，能一眼看穿用户抱怨背后的真实本质，擅长将零散的社区发言转化为开发人员可以直接看的“痛点-功能对应矩阵”。',
    verbose=True,
    allow_delegation=False,
    llm=custom_llm
)

# ==================== 任务 1：情报搜集 ====================
gather_data_task = Task(
    description=(
        '1. 使用搜索工具检索知乎、V2EX、Reddit 等平台上关于“{topic}”相关关键词的最新高热度帖子。\n'
        '2. 选择 2-3 个讨论最激烈的帖子，调用网页爬取工具（ScrapeWebsiteTool）读取帖子的正文及评论内容。\n'
        '3. 汇总整理这些原始发言，去除与主题无关的废话，打包传递给产品分析师。'
    ),
    expected_output='一份包含用户原汁原味吐槽的社区调研原始报告（包含帖子链接和核心言论截图式文字）。',
    agent=data_scout
)

# ==================== 任务 2：深度需求提炼 ====================
analyze_data_task = Task(
    description=(
        '1. 仔细阅读情报搜集专家提供的原始调研报告。\n'
        '2. 提炼出用户关于“{topic}”最痛的 3 个核心场景痛点。\n'
        '3. 针对每个痛点，利用你的产品逻辑设计一个切实可行的“AI 辅助或轻量级功能改进机会”。'
    ),
    expected_output=(
        '一份结构极度清晰的需求提炼日报，包含以下三个板块：\n'
        '1. 【场景与痛点陈述】（说明谁在什么场景下痛苦）\n'
        '2. 【现有解决方案局限性】（为什么市面上的办法他们不用）\n'
        '3. 【产品功能迭代建议】（我们能做点什么，可以用专业互联网黑话进行包装）。'
    ),
    output_file='demand_report.md',  # 运行结果会自动保存并覆盖这个文件
    agent=product_analyst
)

# 4. 组建团队
demand_factory_crew = Crew(
    agents=[data_scout, product_analyst],
    tasks=[gather_data_task, analyze_data_task],
    process=Process.sequential
)

# ==================== 5. 动态交互启动 ====================
if __name__ == "__main__":
    # 在终端让用户直接输入想要研究的主题
    user_topic = input("🔍 请输入你今天想要调研的需求主题（例如：工位久坐、短视频剪辑痛点、新手学英语）：")
    
    if not user_topic.strip():
        user_topic = "工位久坐与健康痛点"  # 如果直接敲回车，则使用默认主题
        
    print(f"\n🚀 社区需求情报工厂开始运转，正在调研主题：【{user_topic}】...")
    
    # 核心：通过 inputs 将变量传入，CrewAI 会自动替换掉代码里所有的 {topic}
    result = demand_factory_crew.kickoff(inputs={"topic": user_topic})

    print("\n==========================================")
    print("🎯 工厂最终交付成果：")
    print(result)
    print("\n[系统提示] 分析结果已同步保存至本地文件 demand_report.md 中！")