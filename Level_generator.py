import random
import os
from openai import OpenAI
import json
import re
from FineTuner import FineTuner
import transformers
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sample_generator import *


# Initialize OpenAI client
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ.get(API_KEY))


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Level:
    def __init__(self, name, difficulty, width, height, num_wall, num_enemies):
        self.name = name
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.num_wall = num_wall
        self.num_enemies = num_enemies



def create_level(level_num, width=20, height=20):
    difficulty = level_num
    num_enemies = min(2 + level_num * 2, (width * height) // 20)
    num_barriers = min(3 + level_num * 3, (width * height) // 15) # Minimum inside barriers=5 / max=20

    return Level(
        name=f"Level-{level_num}",
        difficulty=difficulty,
        width=width,
        height=height,
        num_wall=num_barriers,
        num_enemies=num_enemies
    )




# ADDED: Filter validation functions directly in Level_generator.py
def get_promt_enemy_barrier_count(text: str):
    matches = re.findall(r"Number of (?:enemies|barriers):\s*(\d+)", text)
    if len(matches) == 2:
        num_enemies = int(matches[0])
        num_barriers = int(matches[1])
        return num_enemies, num_barriers
    else:
        return 0, 0


# FIXED: Removed the 'level' parameter since we don't need it for basic validation
def validate_tilemap(tilemap):
    bool_list = []
    PEB_list = []

    num_rows = len(tilemap)
    print(f"Tilemap has {num_rows} rows")

    for i, item in enumerate(tilemap):
        player_count = 0
        enemy_count = 0
        barrier_count = 0

        if len(item) != 20:
            bool_list.append(False)
            player_count = item.count("P")
            enemy_count = item.count("E")
            barrier_count = item.count("B")
            PEB_list.append([player_count, enemy_count, barrier_count])
            continue

        if i == 0:
            if all(char == 'B' for char in item):
                bool_list.append(True)
            else:
                bool_list.append(False)
        elif i == len(tilemap) - 1:
            if all(char == 'B' for char in item):
                bool_list.append(True)
            else:
                bool_list.append(False)
        else:
            if item[0] == "B" and item[-1] == "B":
                bool_list.append(True)
            else:
                bool_list.append(False)

        player_count = item.count("P")
        enemy_count = item.count("E")
        barrier_count = item.count("B")
        PEB_list.append([player_count, enemy_count, barrier_count])

    return bool_list, PEB_list


def get_total_counts(PEB_list):
    if not PEB_list:
        return [0, 0, 0]
    total_player = sum(row[0] for row in PEB_list)
    total_enemy = sum(row[1] for row in PEB_list)
    total_barrier = sum(row[2] for row in PEB_list)
    return [total_player, total_enemy, total_barrier]


def is_playable(tilemap, Level):
    level = Level
    bool_list, PEB_list = validate_tilemap(tilemap)
    #enemies, barriers = get_promt_enemy_barrier_count(prompt_text)
    enemies, barriers = level.num_enemies, level.num_wall

    if not bool_list:
        print(" No validation results")
        return False

    total_counts = get_total_counts(PEB_list)
    print(f"Total barriers: {total_counts[2]}")

    valid_borders = all(bool_list) and (total_counts[2] >= barriers + 76) and (total_counts[2] <= barriers + 86)
    #56 Outer walls in a 20 x 20 map
    # Checks if total barriers lie within a small margin


    if total_counts[0] == 1: # Player check
        valid_player = True
        print("One player found")
    else:
        valid_player = False
        #print(f"Player count: {total_counts[0]} (should be 1)")

    if (total_counts[1] >= enemies) and (total_counts[1] <= enemies + 5):
        # Checking if generated map has more enemies than requested. Can have more than five maximum as well.
        valid_enemy = True
        #print(f"Found {total_counts[1]} enemies (required: {enemies})")
    else:
        valid_enemy = False
        print(f"Enemy count: {total_counts[1]} (should be at least {enemies})")

    is_valid = valid_borders and valid_player and valid_enemy
    if not is_valid:
        """print("Invalid tilemap:")
        for i, row in enumerate(tilemap):
            print(f"Row {i:2}: {row}")"""
        print("Level validation result: FAILED\n")
    else:
        print("Level validation result: PASSED\n")

    return is_valid


# END OF ADDED FILTER FUNCTIONS
def get_prompt(level):
    """Generate a level using OpenAI's API"""
    prompt = f"""
    Generate a top-down 2D game map as a JSON array of strings.

    Specifications:
    - Size: {level.width}x{level.height} tiles
    - Number of barriers: {level.num_wall} inner 'B' tiles + border of 'B's
    - Number of enemies: {level.num_enemies} 'E' tiles
    - Player: 1 'P' tile, placed so it can move
    - Empty spaces: '.' tiles

    Rules:
    1. Map borders must be 'B' (top, bottom, left, right)
    2. Inner 'B' tiles: exactly {level.num_wall}, do not block player
    3. Enemies: exactly {level.num_enemies}
    4. Player: exactly 1 'P', can move
    5. Return only JSON array of strings, no extra text

    Example 5x5:
    ["BBBBB", "B...B", "B.E.B", "B.P.B", "BBBBB"]
    """

    return prompt

def generate_tilemap_with_ai(level):
    prompt = get_prompt(level)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system",
                       "content": "You are a game level designer. Create fun and challenging levels for a 2D top-down game."},
                      {"role": "user", "content": prompt}
                      ],
            temperature=0.7,
            max_tokens=512
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        tilemap = json.loads(content)
        #print(tilemap)
        return prompt, tilemap

    except Exception as e:
        print(f"AI generation failed: {e}. Using fallback level generation.")
        return prompt, generate_fallback_tilemap(level)


def generate_fallback_tilemap(level):
    print("Internal map generation Called")
    w, h = level.width, level.height

    tilemap = []
    for i in range(h):
        if i == 0 or i == h - 1:
            tilemap.append("B" * w)
        else:
            tilemap.append("B" + "." * (w - 2) + "B")

    center_x, center_y = w // 2, h // 2
    row = list(tilemap[center_y])
    row[center_x] = "P"
    tilemap[center_y] = "".join(row)

    enemies_placed = 0
    while enemies_placed < level.num_enemies:
        x = random.randint(1, w - 2)
        y = random.randint(1, h - 2)
        if abs(x - center_x) > 3 or abs(y - center_y) > 3:
            row = list(tilemap[y])
            if row[x] == ".":
                row[x] = "E"
                tilemap[y] = "".join(row)
                enemies_placed += 1

    barriers_placed = 0
    while barriers_placed < level.num_wall:
        x = random.randint(1, w - 2)
        y = random.randint(1, h - 2)
        row = list(tilemap[y])
        if row[x] == ".":
            row[x] = "B"
            tilemap[y] = "".join(row)
            barriers_placed += 1

    if level.difficulty > 3:
        for _ in range(level.difficulty - 3):
            y = random.randint(3, h - 4)
            start_x = random.randint(2, w - 6)
            length = random.randint(3, 5)
            row = list(tilemap[y])
            for x in range(start_x, min(start_x + length, w - 1)):
                if row[x] == ".":
                    row[x] = "B"
            tilemap[y] = "".join(row)

    return tilemap


# FIXED: Keep the original validate_tilemap function with level parameter for backward compatibility
def validate_tilemap_with_level(tilemap, level):
    if not isinstance(tilemap, list):
        return False
    if len(tilemap) != level.height:
        return False
    for row in tilemap:
        if len(row) != level.width:
            return False
        for char in row:
            if char not in ['B', 'E', 'P', '.']:
                return False
    player_count = sum(row.count('P') for row in tilemap)
    if player_count != 1:
        return False
    return True

def get_level_from_llama(level_num):
    l = Level(level_num)
    prompt = get_prompt(l)
    f = FineTuner()
    model, tokenizer = f.load_llama()
    map = f.load_map_from_finetuned_model(prompt, model, tokenizer)
    return map



"""def get_level(level_num):
    level = create_level(level_num)
    feed_prompt = get_prompt(level)
    i = 0
    while i < 10:
        llama_generated_map = get_level_from_llama(feed_prompt)
        if level_num <= 10:
            if is_playable(llama_generated_map, feed_prompt):
                print("LLama Successfully Generated the map!")
                return feed_prompt, llama_generated_map
            else:
                print("LLama failed. Calling GPT")
                prompt, tilemap = generate_tilemap_with_ai(level)
                play_state = is_playable(tilemap, prompt)
                if play_state:
                    print("GPT Successfully Generated the map")
                    return prompt, tilemap
        i += 1
    if i >= 10:
        print("AI couldn't give a valid map for 10 consecetive calls. Generating map internally.")
        return feed_prompt, generate_fallback_tilemap(level)"""

def get_level(level_num):
    s = Sample("hi", "buddy")
    level = create_level(level_num)
    prompt, tilemap = generate_tilemap_with_ai(level)
    play_state = is_playable(tilemap, level)
    if play_state:
        s.increment_success()
        print("GPT Successfully Generated the map at the 1st request")
        return prompt, tilemap
    else:
        s.increment_failure()
        i = 0
        while i < 10:  # Requesting 10 times until a valid map is given
            prompt_new, tilemap_gpt = generate_tilemap_with_ai(level)
            #print(tilemap_gpt)
            if is_playable(tilemap_gpt, level):
                s.increment_success()
                print(f"GPT generation succeeded at {i+2}th attempt")

                return prompt_new, tilemap_gpt
            else:
                s.increment_failure()
            i += 1

        if i >= 10:
            print("AI couldn't give a valid map for 10 consecetive calls. Generating map internally.")
            return prompt, generate_fallback_tilemap(level)


#level = create_level(4)
#print(level.num_wall)
"""
map_n = generate_fallback_tilemap(level)
print(map_n)

prompt, tilemap = get_level(2)
print(tilemap)"""
