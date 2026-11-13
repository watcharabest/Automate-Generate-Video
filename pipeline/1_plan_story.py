import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# กำหนดโครงสร้าง JSON ให้ Gemini ตอบกลับแบบเป๊ะๆ
class Scene(BaseModel):
    scene_number: int
    narration: str = Field(description="บทพากย์ภาษาอังกฤษกระชับ ดึงดูดคนดูฝั่ง US")
    visual_prompt: str = Field(description="Prompt สร้างภาพแนว Isometric 3D Pixel Art, vibrant lighting, 9:16 vertical")
    sfx_cue: Literal["whoosh", "pop", "click", "glitch", "none"] = Field(description="เสียงเอฟเฟกต์ตอนเปิดฉากนี้")

class Episode(BaseModel):
    part_number: int
    title: str
    scenes: List[Scene]

class StoryPlan(BaseModel):
    topic_slug: str = Field(description="ชื่อเรื่องสั้นๆ ภาษาอังกฤษสำหรับตั้งชื่อไฟล์/โฟลเดอร์")
    is_multi_part: bool = Field(description="True หากเนื้อหายาวและควรแบ่งเป็น 2-3 Part, False หากจบใน 1 คลิป")
    episodes: List[Episode]

def generate_story_plan(trend_topic: str) -> dict:
    prompt = f"""
    You are an expert viral YouTube Shorts producer targeting US Tech & AI audiences.
    Topic: {trend_topic}
    
    Task:
    1. Decide if this topic fits best in 1 single 45-second Short, or should be split into 2-3 cliffhanger Parts.
    2. Write punchy, high-retention English narration (short sentences, strong hook in scene 1).
    3. Write detailed 'Isometric 3D Pixel Art' image prompts for each scene.
    4. Assign an appropriate UI/Retro sound effect ('whoosh', 'pop', 'click', 'glitch', or 'none') for each scene transition.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StoryPlan,
            temperature=0.7,
        ),
    )
    return json.loads(response.text)

if __name__ == "__main__":
    sample_topic = "How AI Agents are replacing traditional SaaS apps in 2026"
    plan = generate_story_plan(sample_topic)
    
    os.makedirs("output", exist_ok=True)
    with open("output/story_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        
    print(f"สร้างแผนสำเร็จ! แบ่งเป็น {len(plan['episodes'])} Part -> บันทึกที่ output/story_plan.json")