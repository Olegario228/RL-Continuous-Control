import re
import sys
import random
import itertools
from decimal import Decimal

# --------- #

def parse_decay_steps(input_string):
    # Разделяем строку по запятым
    parts = input_string.split(',')

    # Результат будет списком списков
    result = []

    for part in parts:
        if '-' in part:
            # Для диапазона: находим числа и шаг
            numbers = list(map(int, re.findall(r'\d+', part)))
            if len(numbers) == 2:
                # Если шаг не указан, используем шаг 1
                range_values = list(range(numbers[0], numbers[1] + 1, 1))
            elif len(numbers) == 3:
                # Если шаг указан
                range_values = list(range(numbers[0], numbers[1] + 1, numbers[2]))
            result.append(range_values)
        else:
            # Для одиночного числа: преобразуем в число
            number = int(part.strip())
            result.append([number])

    return result

# ------------- #

def parse_ranges(input_string):
    # Разбиваем входную строку на отдельные диапазоны
    groups = input_string.split(', ')
    result = []

    for group in groups:
        subgroups = group.split(' ')
        subresult = []
        for subgroup in subgroups:
            numbers = list(map(int, re.findall(r'\d+', subgroup)))
            if '-' in subgroup:  # Если это диапазон
                range_values = list(range(numbers[0], numbers[1] + 1, numbers[2] if len(numbers) > 2 else 1))
                subresult.append(range_values)
            else:  # Если это одно число
                subresult.append(numbers)
        result.append(subresult)
    return result

def generate_collection_combinations(parsed_ranges):
    for product in itertools.product(*[itertools.product(*subgroup) for subgroup in parsed_ranges]):
        yield list(map(list, product))

def generate_decay_combinations(steps):
    # Предполагаем, что первый список содержит только одно значение
    first_value = steps[0][0]

    # Второй и третий списки могут иметь разную длину, поэтому мы берем максимальную длину из них
    max_length = max(len(steps[1]), len(steps[2]))

    # Генерируем комбинации
    for i in range(max_length):
        # Получаем значения из второго и третьего списков, учитывая их длину
        second_value = steps[1][min(i, len(steps[1]) - 1)]
        third_value = steps[2][min(i, len(steps[2]) - 1)]

        # Выводим комбинацию
        yield [first_value, second_value, third_value]

# Тестирование функции

def generate_sequential_combinations(parsed_ranges):
    max_steps = min(len(r) for group in parsed_ranges for r in group)
    
    for step in range(max_steps):
        current_combination = []
        for group in parsed_ranges:
            current_group = []
            for range_values in group:
                # Добавляем в группу минимальное значение из range_values, увеличенное на step
                # Но не превышающее максимальное значение в range_values
                current_value = min(range_values[0] + step, range_values[-1])
                current_group.append(current_value)
            current_combination.append(current_group)
        yield current_combination

def parse_d_factor_range(range_string):
    numbers = [Decimal(n) for n in re.findall(r"\d+\.\d+", range_string)]
    
    if len(numbers) == 3:
        start, end, step = numbers
    elif len(numbers) == 2:
        start, end = numbers
        step = Decimal('0.0001')  # Значение шага по умолчанию

    return start, end, step

def generate_d_factor_values(start, end, step):
    current = start
    while current <= end:
        yield current
        current += step
#-------------- #

bestNSteps = sys.maxsize

params = {
    "decaySteps": {
        "variants": "0, 5000-10000(500), 8000-16000(1000)",
        "brut": False,  # Если True, то проходит все комбинации. Если False, то только пары комбинаций
    },
    "controlValues": {
        "variants": "2-4(1) 4-10(1), 0-0 0-9(1), 0-0 0-0",
        "brut": False,
    },
    "dFactor": {
        "variants": "0.0005-0.0010(0.0001)",
    },
    "decayType": {
        "variants": "exp", #, lin, quad, x^5, discrete, sqrt
    },
  }

parsed_decay_values = parse_decay_steps(params["decaySteps"]["variants"])
parsed_control_values = parse_ranges(params["controlValues"]["variants"])
parsed_range = parse_d_factor_range(params["dFactor"]["variants"])
decay_types = params["decayType"]["variants"].split(", ")

# Генерация комбинаций для decaySteps
if params["decaySteps"]["brut"]:
    decay_combinations = itertools.product(*parsed_decay_values)
else:
    decay_combinations = generate_decay_combinations(parsed_decay_values)


if params["controlValues"]["brut"]:
    control_combinations = generate_collection_combinations(parsed_control_values)
else:
    control_combinations = generate_sequential_combinations(parsed_control_values)

control_combinations = list(control_combinations)
decay_combinations = list(decay_combinations)

# Перебор всех комбинаций
for control_combination in control_combinations:
  for decay_combination in decay_combinations:
    for d_factor in generate_d_factor_values(parsed_range[0], parsed_range[1], parsed_range[2]):
      for decay_type in decay_types:
        randomNStep = random.randint(1000, 100000)
        print("randomNStep: ", randomNStep, " and combination: ", decay_combination, "and controlValue: ", control_combination, " and dFactor: ", d_factor, " and decayType: ", decay_type)
        if randomNStep <= bestNSteps:
            bestNSteps = randomNStep

print("bestNSteps: ", bestNSteps)
