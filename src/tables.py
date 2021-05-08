# generate the tables for the report
import pandas as pd
import numpy as np
import datetime
import os
import re
from typing import Dict

res_dir = "../res"
target_renaming = {
    'TOT_HOSP': 'total hospitalizations',
    'NEW_HOSP': 'new hospitalizations',
}

model_renaming = {
    'assemble': 'Assembled',
    'custom_linear_regression': 'Linear Regression',
    'dense_model': 'Dense',
    'encoder_decoder': 'Encoder Decoder',
    'baseline': 'Baseline'
}
len_dates = 16  # len of the format 'YYYY-MM-DD-HH:MM'


def weights_assemble_table(info_file: str):
    """
    generate the table for the weights used by an assembly model
    :param info_file: name of the info file, with the extension
    """
    df_weights = pd.DataFrame()
    float_pattern = '(\d+.\d+)'
    walk_pattern = 'weights of walk (\d+) = |means of weights = '
    n_forecast = 0
    with open(f'{res_dir}/{info_file}', 'r') as file:
        for line in file:
            search_obj_walk = re.search(walk_pattern, line)
            if search_obj_walk is not None:
                search_obj = re.findall(float_pattern, line)
                data_weights = {}
                for i in range(len(search_obj)):
                    data_weights[f't+{i+1}'] = [float(search_obj[i])]
                if search_obj_walk.lastindex is not None:
                    walk = f'walk {search_obj_walk.group(1)}'
                else:
                    walk = 'mean'
                df = pd.DataFrame(data={**data_weights, **{'Horizon': walk}}).set_index('Horizon')
                df_weights = df_weights.append(df)
    df_weights = df_weights.transpose()
    caption = 'Weights chosen by the assembler on each horizon'
    label = 'tab:weights_assembler'
    float_format = '.02f'
    table = df_weights.to_latex(caption=caption, label=label, float_format=f'%{float_format}')
    table = table.replace("\\begin{table}", "\\begin{table}[H]")
    tex_filename = f'{res_dir}/{info_file.replace(".txt", "")}_weights.tex'
    table = f'%{info_file.replace(".txt", "")}\n' + table
    with open(tex_filename, 'w') as file:
        file.write(table)
    return table


def extract_info_file(info_file: str) -> Dict[str, str]:
    """
    extract the parameters of an info file and return the dict of relevant informations related to it
    :param info_file: complete path to the info file
    """
    type_dataset = 'European countries'
    with open(info_file, 'r') as file:
        for line in file:
            search_obj = re.search('nb init regions = (\d+), nb augmented regions = (\d+)', line)
            if search_obj is not None:
                n_init_regions = search_obj.group(1)
                n_augmented_regions = search_obj.group(2)
            else:
                search_obj = re.search("'FR-A'", line)
                if search_obj is not None:
                    type_dataset = 'French regions and Belgium'
    return {
        'type_dataset': type_dataset,
        'n_init_regions': n_init_regions,
        'n_augmented_regions': n_augmented_regions,
    }


def highlight_table(table: str, min_format: str = 'textbf', max_format: str = None, float_format: str = '.02f') -> str:
    """
    highlight the values of a LaTeX table. The data columns to hightlight must begin in the format 't+i' with i being an
    integer
    :param table: latex table to edit, given as string
    :param min_format: format to use for the minimum values of each row
    :param max_format: format to use for the maximum values of each row
    :param float_format: format used to write the float values
    :return edited LaTeX table, where the min/max values are highlighted
    """
    table_bold = []
    first_row = True
    for i, line in enumerate(table.split('\\\\')):
        if i == 0:
            table_bold.append(line.replace("\\begin{table}", "\\begin{table}[H]"))
            continue
        search_obj = re.search('t\+(\d+)', line)
        if search_obj is None:  # not a data row
            table_bold.append(line)
        else:
            # extract the numbers in the string
            numbers = [float(i) for i in re.findall('(\d+.\d+)', line)]
            min_number = min(numbers)
            max_number = max(numbers)
            if first_row:
                line_bold = f'\\midrule\n{search_obj.group()}  '
                first_row = False
            else:
                line_bold = f'{search_obj.group()}  '
            for value in numbers:
                if value == min_number and min_format is not None:
                    line_bold += f'& \\{min_format}{{{ value:{float_format}}}}  '
                elif value == max_number and max_format is not None:
                    line_bold += f'& \\{max_format}{{{ value:{float_format}}}}  '
                else:
                    line_bold += f'& { value:{float_format}}  '
            table_bold.append(line_bold)
    return str.join('\\\\\n', table_bold)


def walk_table_assembler_header(df: pd.DataFrame, first_walk_nb: int = 1, last_walk_mean: bool = True):
    """
    write the table of error for the assembly model on each walk
    :param nu
    :param last_walk_mean: if True, the last walk header will be written as 'mean'. Otherwhise, the number will be added
    """
    # each walk is written as a multicolumn
    len_cols = len(df.columns)
    nb_walks = int(len_cols / 3)
    header = '\n '
    for i in range(first_walk_nb, nb_walks + first_walk_nb - 1):
        header += f'& \\multicolumn{{3}}{{c}}{{walk {i}}} '
    header += '& \\multicolumn{3}{c}{mean} '
    header += '\n\\\\\n'
    for i in range(1, nb_walks + 1):  # 2-4, 5-7
        header += f'\\cmidrule(lr){{{i*3-1}-{i*3+1}}} '
    header += '\nHorizon '
    for i in range(nb_walks):
        header += f'& $\\hat{{y}}_{{t+i}}^H$ & $\\hat{{y}}_{{t+i}}^T$ & $\\hat{{y}}_{{t+i}}$ '
    header += '\\\\\n'
    return header


def walk_table(model_file: str, error: str) -> str:
    """
    write the table of error for the given model on each walk
    :param model_file: model file, without the extension
    :param error: error to use, must be registered as an error in the csv files
    """
    search_obj = re.search(f'(TOT_HOSP|NEW_HOSP)', model_file)
    target = search_obj.group(1)
    df = pd.read_csv(f'{res_dir}/{model_file}.csv').rename(columns={'name': 'walk'})
    len_error = len(error)
    n_forecast = len([col for col in df.columns if col[:len_error] == error])
    columns_errors = [f'{error}(t+{i})' for i in range(1, n_forecast + 1)]
    renaming_horizon = {col: f't+{i+1}' for i, col in enumerate(columns_errors)}
    renaming = {**renaming_horizon, **{'walk': 'Horizon'}}
    df = df.rename(columns=renaming).set_index('Horizon')[renaming_horizon.values()]
    df = df.transpose()
    info_file = f'{res_dir}/{model_file}.txt'
    model_name = re.search('_get_(.*)_(TOT_HOSP|NEW_HOSP)', model_file).group(1)
    info = extract_info_file(info_file)
    float_format = '.02f'
    label = f'tab:{error}_walk_{model_name}'
    caption = f'{error} on each walk when predicting {target_renaming[target]} for the model, for up to {n_forecast} ' \
              f'horizons. The mean over all walks is also reported. Boldface indicates the best performance on each row. ' \
              f'The dataset covered the {info["type_dataset"]}, composed of {info["n_init_regions"]} initial regions ' \
              f'and {info["n_augmented_regions"]} augmented regions '
    if 'assemble' in model_file:
        nb_walks = int(len(df.columns) / 3) - 1
        shown_walks = [f'{i}' for i in range(nb_walks-1, nb_walks+1)]
        # uses only the last 2 walks and the mean
        columns_shown = [col for col in df.columns if col[5] in shown_walks] + [i for i in df.columns[-3:]]
        df = df[columns_shown]
        table = df.to_latex(float_format=f'%{float_format}', caption=caption, label=label)
        header = walk_table_assembler_header(df, first_walk_nb=nb_walks-1)
        # TODO highlight table for assembly (best value amongst y^H, y^T and y)
        header_start = table.index('\\toprule') + len('\\toprule')
        header_end = table.index('\\midrule')
        table = table[:header_start] + header + table[header_end:]
    else:
        table = df.to_latex(float_format=f'%{float_format}', caption=caption, label=label)
    table_bold = highlight_table(table, float_format=float_format)
    tex_filename = f'{res_dir}/{model_file}_{error}.tex'
    table_bold = f'%{model_file}_{error}\n' + table_bold
    with open(tex_filename, 'w') as file:
        file.write(table_bold)
    return table_bold


def mae_walk_table(model_file: str) -> str:
    """
    write the table of MAE in .tex format for all models showm
    :param model_file: model file, without the extension
    """
    return walk_table(model_file, 'MAE')


def mse_walk_table(model_file: str) -> str:
    """
    write the table of MSE in .tex format for all models showm
    :param model_file: model file, without the extension
    """
    return walk_table(model_file, 'MSE')


def comparison_table(date: str, error: str) -> str:
    """
    write the comparison table for all models on the date given
    :param date: beginning of the files to search for, in the format YYYY-MM-DD-HH:MM
    :param error: error to use, must be registered as an error in the csv files
    """
    error_df = pd.DataFrame()
    columns_errors = []
    target = ''
    len_error = len(error)
    for file in os.listdir(res_dir):
        # get the average of each model
        search_obj = re.search(f'{date}_get_(.*)_(TOT_HOSP|NEW_HOSP).csv', file)
        if search_obj is not None:
            df = pd.read_csv(f'{res_dir}/{file}').rename(columns={'name': 'model', 'walk': 'model'}).set_index('model')
            # extract the error
            n_forecast = len([col for col in df.columns if col[:len_error] == error])
            if not columns_errors:  # first csv file found
                columns_errors = [f'{error}(t+{i})' for i in range(1, n_forecast+1)]
                info_file = f'{res_dir}/{file.replace(".csv", ".txt")}'
                target = search_obj.group(2)
            else:  # verify that all models covered the same horizon
                assert n_forecast == len(columns_errors), 'all models should have errors on the same horizon'
            entry = df.iloc[-1][columns_errors]#.reset_index()  # get the last row
            entry['model'] = model_renaming[search_obj.group(1)]
            #entry.set_index('model')
            error_df = error_df.append(entry)
    if not columns_errors:
        raise Exception('files not found')
    info = extract_info_file(info_file)
    renaming_horizon = {col: f't+{i+1}' for i, col in enumerate(columns_errors)}
    renaming = {**renaming_horizon, **{'model': 'Horizon'}}
    error_df = error_df.rename(columns=renaming).set_index('Horizon')[renaming_horizon.values()]
    # write to LaTeX format
    error_df = error_df.transpose()
    float_format = '.02f'
    caption = f'{error} when predicting {target_renaming[target]} for the different models, for up to {n_forecast} ' \
              f'horizons. Boldface indicates the best performance on each row. The dataset covered the {info["type_dataset"]}, '\
              f'composed of {info["n_init_regions"]} initial regions and {info["n_augmented_regions"]} augmented regions '
    label = f'tab:{error}_comparison'
    table = error_df.to_latex(float_format=f'%{float_format}', caption=caption, label=label)
    # use bold format for the best values
    table_bold = highlight_table(table, float_format=float_format)
    tex_filename = f'{res_dir}/{date}_{error}_{target}.tex'
    table_bold = f'%{date}_{error}_{target}\n' + table_bold
    with open(tex_filename, 'w') as file:
        file.write(table_bold)
    return table_bold


def mae_comparison_table(date: str) -> str:
    """
    write the table of MAE in .tex format for all models showm
    :param date: beginning of the files to search for, in the format YYYY-MM-DD-HH:MM
    """
    return comparison_table(date, 'MAE')


def mse_comparison_table(date: str) -> str:
    """
    write the table of MSE in .tex format for all models showm
    :param date: beginning of the files to search for, in the format YYYY-MM-DD-HH:MM
    """
    return comparison_table(date, 'MSE')


def generate_all_tables_date(date: str):
    # generate the comparison tables
    mae_comparison_table(date)
    mse_comparison_table(date)
    # generate the walks tables
    len_date = len(date)
    for file in os.listdir(res_dir):
        if file[:len_date] == date and file[-4:] == '.csv':
            mae_walk_table(file[:-4])
            mse_walk_table(file[:-4])
            if 'assemble' in file:
                weights_assemble_table(f'{file[:-4]}.txt')


def generate_all_tables():
    """
    generate all possible tables in the res folder
    """
    list_date = set()
    for file in os.listdir(res_dir):
        date = file[:len_dates]
        if date not in list_date:
            list_date.add(date)
            generate_all_tables_date(date)


if __name__ == '__main__':
    generate_all_tables()
    # weights_assemble_table('2021-04-24-18:56_get_assemble_TOT_HOSP.txt')
