# ====================================================================================================
#                      разработка скоринговой функции в соответствии с требованиями
# ====================================================================================================

from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
import numpy as np

DEFAULT_CONFIG = {
    # --- Лимиты для расчета IMC (ИЗ ИСХОДНЫХ ДАННЫХ) ---
    'limit_bertz': 1600.0,   # Эмпирически, т.к. MW max=800 (Bertz ~ 2*MW для сложных)
    'limit_stereo': 4.0,     # max = 4.0
    'limit_rotatable': 10.0, # max = 10.0
    'limit_sp3': 0.9,        # max = 0.9
    'limit_aromatic': 5.0,   # max = 5.0

    # --- Лимиты для RBS ---
    'limit_steps': 10.0,     # Ожидаемый макс. шагов синтеза
    # limit_state не нужен, так как он в диапазоне [0; 1]

    # --- Веса для IMC (структура) ---
    # Положительные веса усложняют, отрицательные упрощают
    'w_bertz': 0.40,         # Общая сложность
    'w_stereo': 0.30,        # Стереохимия (сильно усложняет)
    'w_rotatable': 0.10,     # Гибкость
    'w_sp3': 0.10,           # 3D сложность
    'w_aromatic': -0.10,     # Ароматика обычно упрощает химию (стабильные циклы)

    # --- Веса для RBS (Путь) ---
    'w_rbs_steps': 0.70,      # 70% сложности пути - это количество шагов
    'w_rbs_stock': 0.20,      # 20%: Доступность билдинг-блоков (реализуемость)
    'w_rbs_state': 0.10,      # 10% сложности - это неуверенность модели
    

    # --- Веса для ИТОГА (60% / 40%) ---
    'mix_w_imc': 0.60,      # 60% итоговой оценки - это структура
    'mix_w_rbs': 0.40,      # 40% итоговой оценки - это путь
    
    # --- Штрафы (если путь не найден) ---
    'penalty_base': 60.0,    # Если пути нет, старт с 60 баллов. То есть скорость сразу высокая
    'penalty_factor': 0.40   # Плюс 40% от сложности структуры
}

def calculate_synth_score(finder, mol: Chem.Mol, config: dict=None) -> float:

    if config is None:
        config = DEFAULT_CONFIG

    # ===== 2. Расчет вклада от IMC (только структурные факторы) =======
    try:
        # если свойства  'Molecular Complexity' нет в исходных данных
        mol_complexity = float(mol.GetProp('Molecular Complexity'))
    except Exception: 
        # то, считаем с помощью RDKit
        mol_complexity = Descriptors.BertzCT(mol)


    def get_prop(name):
        # Функция для получения свойств из SDF файла
        # Но можно было бы рассчитать их с помощью RDKit
        try: 
            return float(mol.GetProp(name))
        except Exception as e: 
            return 0.0

    imc_properties = {
        "MolecularComplexity": mol_complexity, 
        "Stereo Centers": get_prop("Stereo Centers"),
        "Rotatable Bonds": get_prop("Rotatable Bonds"), 
        "sp3-Carbon Fraction": get_prop("sp3-Carbon Fraction"),
        "Aromatic Rings": get_prop("Aromatic Rings"),
    }

    # Номрализация -> [0; 1]
    n_bertz = np.clip(imc_properties["MolecularComplexity"] / config['limit_bertz'], 0, 1)
    n_stereo = np.clip(imc_properties["Stereo Centers"] / config['limit_stereo'], 0, 1)
    n_rotat = np.clip(imc_properties["Rotatable Bonds"] / config['limit_rotatable'], 0, 1)
    n_sp3 = np.clip(imc_properties["sp3-Carbon Fraction"] / config['limit_sp3'], 0, 1)
    n_arom = np.clip(imc_properties["Aromatic Rings"] / config['limit_aromatic'], 0, 1)

    # Расчет взвешанной суммы
    wsum_imc = (
        (n_bertz * config['w_bertz']) +
        (n_stereo * config['w_stereo']) +
        (n_rotat * config['w_rotatable']) +
        (n_sp3 * config['w_sp3']) +
        (n_arom * config['w_aromatic'])
    )

    imc_score = max(0.0, wsum_imc) * 100.0

    # === 3. Запуск AiZynthFinder и расчет вклада от RBS ===
    finder.target_smiles = Chem.MolToSmiles(mol)
    finder.tree_search()
    finder.build_routes()

    status_log = ""
    details_log = "N/A"
    rbs_details = [] # Для логгера

    # -----------------------------------------
    routes = getattr(finder.analysis, 'reaction_routes', [])
    final_score = 0.0

    if routes:
        status_log = "Путь найден ✔️"
        # best_route = finder.analysis.reaction_routes[0]
        best_route = routes[0]
        scores = best_route.scores

        # 1. Шаги (Steps) - Сложность процесса
        val_steps = getattr(scores, "NumberOfReactionsScorer", 0.0)
        n_steps = np.clip(val_steps / config['limit_steps'], 0, 1)

        # 2. Сток (Stock) - Доступность реагентов
        # Считаем, сколько листьев (precursors) есть в наличии
        val_stock_count = getattr(scores, "NumberOfPrecursorsInStockScorer", 0.0)
        total_precursors = max(1, len(best_route.leafs())) # Защита от деления на 0
        
        # Ratio: 1.0 (все есть) -> 0.0 (ничего нет).
        # Нам нужна сложность: (1 - Ratio). 
        # Если все есть -> 0 сложность. Если ничего нет -> 1 сложность.
        n_stock_diff = np.clip(1.0 - (val_stock_count / total_precursors), 0, 1)

        # 3. Уверенность (State) - Риски
        val_state = getattr(scores, "StateScorer", 1.0)
        n_state_diff = np.clip(1.0 - val_state, 0, 1)

        details_log = f"Шагов: {int(val_steps)}, Билдинг-блоки: {int(val_stock_count)}/{total_precursors}"

        wsum_rbs = (
            (n_steps * config['w_rbs_steps']) +
            (n_stock_diff * config['w_rbs_stock']) +
            (n_state_diff * config['w_rbs_state'])
        )
        rbs_score = wsum_rbs * 100.0
        
        # Финальный расчет
        final_score = (imc_score * config['mix_w_imc']) + (rbs_score * config['mix_w_rbs'])

        # для лога
        rbs_details = (f"Steps: {n_steps*100:.0f}%, "
                           f"StockDiff: {n_stock_diff:.2f}, "
                           f"StateDiff: {n_state_diff:.2f}")

        # imc
        contrib_imc = imc_score * config['mix_w_imc']

        contrib_rbs = rbs_score * config['mix_w_rbs']
    
    else:
        # путь не найден, применяем штраф
        status_log = "Путь НЕ найден"
        details_log = "Применяем штраф"
        
        # Логика: Базовый штраф + Часть сложности молекулы
        final_score = config['penalty_base'] + (imc_score * config['penalty_factor'])
        
        # В RBS score пишем 0 или сам штраф для отладки, но на final_score это не влияет напрямую по формуле выше
        rbs_score = 0.0 
        rbs_details.append(f"Штраф: {config['penalty_base']})")

        contrib_imc = imc_score * config['penalty_factor'] # и IMC рассчитывается по-другому
        contrib_rbs = final_score - (imc_score * config['penalty_factor']) # Условный вклад штрафа

    # Гарантируем результат 0-100
    final_score = max(0.0, min(final_score, 100.0))

    # --- Строка для лога ---
    smiles = Chem.MolToSmiles(mol)
    imc_details_list = [f"{k}: {v:.1f}" for k, v in imc_properties.items() if v > 0]
    imc_str = ", ".join(imc_details_list)
    rbs_str = ", ".join(rbs_details)

    log_message = (
        f"{smiles} -> SynthScore: {final_score:.2f} | {status_log} | {details_log}\n"
        f"    IMC (Score: {imc_score:.1f} -> Вклад 60%: {contrib_imc:.1f}): {imc_str}\n"
        f"    RBS (Score: {rbs_score:.1f} -> Вклад 40%: {contrib_rbs:.1f}): {rbs_str}"
    )

    logging.info(log_message)
    
    return final_score