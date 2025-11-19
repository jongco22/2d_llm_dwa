import openai
import json
import os
import time

# OpenAI API 키 설정 (본인의 키로 변경)
openai_api_key = os.getenv("OPENAI_API_KEY")

sysprompt = """You are a path planning assistant for mobile robots navigating using the Dynamic Window Approach (DWA).
Your task is to generate a set of intermediate waypoints to guide the robot from a start point to a goal point while
avoiding local minima and obstacles, and while ensuring that the generated path is as short as possible.

### Environment Information:
1. The map is a 2D grid where the bottom-left corner is (0,0).
2. The X-axis increases from left to right.
3. The Y-axis increases from bottom to top.
4. Obstacles are defined as a list of (x, y, width, height), where:
   - (x, y) represents the **bottom-left corner** of the obstacle.
   - width and height define the size of the obstacle.
5. The robot follows a sequence of waypoints in the format [(x, y), (x, y), ...].

### Requirements:
- The path must be **collision-free** and **avoid all obstacles**.
- The waypoints should form a **smooth**, **optimal**, and **short** path to the goal.
- Selecting appropriate points on the corners of obstacles can help generate the shortest possible path.
- The fewer the waypoints, the better; avoid unnecessary intermediate waypoints.
- Do not generate any waypoint that is beyond the goal.
- If the distance from the current waypoint to the next waypoint is longer than the distance from the current waypoint to the final goal, do not select it.
- Avoid local minima that could trap the robot.
- Ensure the path is **realistic** for an actual mobile robot.
- Return only a valid JSON object with the "Generated Path" list and no additional text.
"""

def get_waypoints(grid_size, start, goal, obstacles):
    start_time = time.time()
    """
    주어진 시작점과 목표점, 그리고 장애물 정보를 바탕으로 LLM을 이용해
    chain-of-thought 과정을 포함한 경로를 생성하는 함수입니다.
    
    각 Iteration마다 LLM은 다음 정보를 포함해야 합니다:
    - 현재 Iteration의 기준 점
    - 장애물이나 경로상의 문제점을 분석한 Thought
    - 우회 또는 진행을 위한 Selected Point
    - 그 단계에 대한 평가(Evaluation)
    
    최종적으로 "Generated Path: [[x1, y1], [x2, y2], ...]" 형태의 경로를 반환합니다.
    
    :param start: [x, y] 형태의 시작 좌표
    :param goal: [x, y] 형태의 목표 좌표
    :param obstacles: [[x, y, width, height], ...] 형태의 장애물 리스트
    :return: LLM의 chain-of-thought 전체 응답 문자열
    
    """

    user_prompt = f"""
    I will provide the environment details as a JSON object.
    You must return only a valid JSON object with a "waypoints" list.
    Do not include any explanation or extra text.

    Given the following input information, generate a collision-free path from the start point to the goal point 
    while navigating around the obstacles. Obstacles are defined as [x, y, width, height]. The final output must conclude with "Generated Path: [[x1, y1], [x2, y2], ...]".

    Additionally, ensure that the first element of the waypoints is the start point {start} and the last element is the goal point {goal}.

    Input:
    grid_size = {grid_size}
    Start Point = {start}
    Goal Point = {goal}
    Obstacles = {obstacles}
    Please adhere to the following instructions:
    - For each iteration, specify the current evaluation point.
    - Provide a detailed chain-of-thought explanation (Thought) about obstacles or necessary adjustments.
    - Provide the Selected Point for the next move.
    - Provide an Evaluation of the chosen point's effectiveness.
    
    Example 1:
    Environment Details:
        grid_size = (20, 20)
        start = (3, 7)
        goal = (1, 19)
        obstacles = [(6, 5, 1, 6), (6, 5, 7, 1), (12, 5, 1, 5), (18, 3, 1, 5), (9, 10, 1, 4), (1, 13, 13, 1), (2, 18, 4, 1)]
    -First iteration on [3,7]
    Thought: The obstacle at [6, 5, 1, 6] blocks the direct path to the goal. To navigate around it, we should move to upper-left corner of the obstacles.
    Selected Point: [5, 11]
    Evaluation: The selected point [5, 11] effectively bypasses the obstacle, positioning us at its corner and maintaining progress toward the goal without encountering additional obstacles.
    -Second iteration on [5, 11]
    Thought: The obstacle at [9, 10, 1, 4] blocks the direct path to the goal. To navigate around it, we should move to the lower-left corner of the obstacle.
    Selected Point: [8, 9]
    Evaluation: The selected point [8, 9] effectively bypasses the obstacle, positioning us at its corner and maintaining progress toward the goal without encountering additional obstacles.
    -Third iteration on [8, 9]
    Thought: When the robot is at [8,9], it is trapped inside a U-shaped obstacle formed by three obstacles: [6, 5, 1, 6], [6, 5, 7, 1], and [12, 5, 1, 5]. To avoid getting stuck in a local minima, an intermediate waypoint is set near the obstacles' edge at (14,13).
    Selected Point: [14, 13]
    Evaluation: The selected point [14, 13] successfully navigates around the U-shaped obstacle, allowing the robot to continue toward the goal without getting trapped.
    -Fourth iteration on [14, 13]
    Thought: The path to the goal is clear from this position, allowing a direct move to the goal.
    Selected Point: [1, 19]
    Evaluation: The path to the goal is clear from here, allowing a direct move to the goal.

    Generated Path: [[3, 7], [5, 11], [8, 9], [14, 13], [1, 19]]

    Example 2:
    Environment Details:
        grid_size = (30, 30)
        start = (28, 28)
        goal = (23, 5)
        obstacles = [(24, 25, 6, 1), (28, 2, 1, 21), (5, 15, 1, 8), (5, 14, 20, 1), (5, 23, 5, 1)]    
    -First iteration on [28,28]
    Thought: The obstacle at [24, 25, 6, 1] blocks the direct path to the goal. To navigate around it, we should move to left side of the obstacles.
    Selected Point: [23, 25]
    Evaluation: The selected point [23, 25] effectively bypasses the obstacle, positioning us at its corner and maintaining progress toward the goal without encountering additional obstacles.
    -Second iteration on [23, 25]
    Thought: The goal point is located below the obstacle at [5, 14, 20, 1]. To reach the goal, we should set the next waypoint at the right side of this obstacle.
    Selected Point: [26, 14]
    Evaluation: The selected point [26, 14] effectively bypasses the obstacle, positioning us at its corner and maintaining progress toward the goal without encountering additional obstacles.
    -Third iteration on [23, 5]
    Evaluation: The path to the goal is clear from here, allowing a direct move to the goal.

    Generated Path: [[28, 28], [23, 25], [26, 14], [23, 5]]
    
    Please generate the complete chain-of-thought reasoning text including the final generated path.
    """

    

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )


    def extract_waypoints(response_content):
        """
        응답 문자열에서 'Generated Path:' 이후의 JSON 형식의 부분을 추출합니다.
        """
        try:
            json_str = response_content.split("Generated Path:")[-1].strip()
            waypoints = json.loads(json_str)
            return waypoints
        except Exception as e:
            print("JSON 파싱 또는 추출 중 오류 발생:", e)
            raise e

    print(f'response: {response}')
    response_content = response["choices"][0]["message"]["content"]
    waypoints = extract_waypoints(response_content)
    print("Generated Waypoints:", waypoints)
    end_time = time.time()
    print(f'api_total_time: {end_time - start_time:.2f} seconds')
    return waypoints['waypoints']

if __name__ == "__main__":
    grid_size = (30, 30)
    start = (2, 25)
    goal = (28, 5)
    obstacles = [(5, 20, 3, 3), (15, 15, 4, 4), (25, 10, 5, 2)]

    waypoints = get_waypoints(grid_size, start, goal, obstacles)
    print("Generated Waypoints:", waypoints)
