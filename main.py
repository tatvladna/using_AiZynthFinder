# ==========================================================================================================
#                                                  точка входа
# ==========================================================================================================

# ---------------------------------- внешние библиотеки -------------------------------
from aizynthfinder.aizynthfinder import AiZynthFinder
from dotenv import load_dotenv
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import datetime
import logging
import time
import os
# ----------------------------------- локальные импорты --------------------------------
from src.scoring import calculate_synth_score
from utils.utils import setup_logging

def main():

    load_dotenv()
    
    input_sdf = os.getenv("INPUT_SDF_PATH")
    output_sdf = os.getenv("OUTPUT_SDF_PATH")
    log_path = os.getenv("LOG_PATH")

    output_path = Path(output_sdf)
    output_path.parent.mkdir(exist_ok=True)
    log_path = Path(log_path) # logger будет сохраняться в одну папку с выходным файлом SDF
    
    setup_logging(log_path) # logger

    logging.info("🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​​ START ​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​🍀​")
    try:

        # см -> https://github.com/MolecularAI/aizynthfinder/blob/master/aizynthfinder/context/config.py
        # оставляем базовые настройки
        aizynthfinder = AiZynthFinder(configfile="./aizynth-data/config.yml") # передаем путь конфига в AiZynthFinder
        logging.info("С конфигом все ок 👍​")
        aizynthfinder.expansion_policy.select(["uspto", "ringbreaker"])
        aizynthfinder.filter_policy.select("uspto")
        aizynthfinder.stock.select("zinc")
        logging.info("выбраны рабочие политики​")
    except Exception as e:
        logging.error(f"Ошибка при передаче конфига в AiZynthFinder: {e}")
        return

    suppl = Chem.SDMolSupplier(input_sdf)
    logging.info(f"Всего молекул в наборе: {len(suppl)}")
    writer = Chem.SDWriter(str(output_path))
    logging.info(f"Обработка {input_sdf} ... ")
    start_time = time.monotonic()


    processed_count = 0
    error_count = 0
    skipped_count = 0

    for mol in tqdm(suppl, desc="Расчет SynthScore 💤​💤​💤​"):
        if mol is None: # обязательно это прописываем, так как анализ показал наличие NaN
            skipped_count += 1
            logging.warning("Обнаружена некорректная запись в SDF")
            continue 

        smiles = Chem.MolToSmiles(mol)
        try:
            score = calculate_synth_score(aizynthfinder, mol)
            mol.SetDoubleProp("SynthScore", score)
            # logging.info(f"{smiles} -> SynthScore = {score:.4f}")
            processed_count += 1
        except Exception as e:
            logging.error(f"{smiles} -> ОШИБКА: {e}")
            mol.SetDoubleProp("SynthScore", -1) # если будет ошибка, то score = -1
            error_count += 1
        writer.write(mol)
    writer.close()

    end_time = time.monotonic()
    total_time = end_time - start_time
    duration = datetime.timedelta(seconds=total_time)
    formatted_duration = str(duration).split('.')[0]
    
    logging.info(f"Всего молекул в наборе: {len(suppl)}")
    logging.info(f"Успешно обработано: {processed_count}")
    logging.info(f"Пропущено (NaN): {skipped_count}")
    logging.info(f"Завершено с ошибкой (score=-1): {error_count}")

    logging.info(f"Обработка завершена.\nОбщее время выполнения:\nСекунды -> {total_time:.2f}\nМинуты -> {total_time/60:.2f}\nЧасы -> {formatted_duration}")
    logging.info("☃️​❄️​☃️​❄️​☃️​❄️​☃️​❄️☃️​❄️​☃️​❄️​☃️​❄️​​ FINISH ☃️​❄️​☃️​❄️​☃️​❄️​☃️​❄️​☃️​❄️☃️​❄️​☃️​❄️​​")


if __name__ == "__main__":
    main()