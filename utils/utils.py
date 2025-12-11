# ===================================================================================================
#                                           логирование
# ===================================================================================================
import logging
import pandas as pd

def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path,
        filemode='w'
    )

# Настройки pandas`a
def start():
    print("Настройки pandas`a применены")
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,
            'max_rows': 35,
            'max_seq_items': 50,
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None
        }
    }
    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)