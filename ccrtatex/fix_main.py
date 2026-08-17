import sys

file_path = r'c:\DR2\ECCL-Tabular-NeuroRejected\ccrtatex\main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Preamble Fix
for i, line in enumerate(lines):
    if '\\usepackage{array}' in line:
        lines.insert(i+1, '\\usepackage{caption}\n')
        break

# 2. Double-tilde fix
for i, line in enumerate(lines):
    if '(GCE)~~\\cite{zhang2018generalized}' in line:
        lines[i] = line.replace('(GCE)~~\\cite{zhang2018generalized}', '(GCE)~\\cite{zhang2018generalized}')
        break

def process_table(label_substr, to_star=True, stretch=None, colsep=None, wrap_resize=False):
    # Find the line with the label
    label_idx = -1
    for i, line in enumerate(lines):
        if label_substr in line:
            label_idx = i
            break
    if label_idx == -1: return
    
    # Find \begin{table... backwards
    begin_idx = -1
    for i in range(label_idx, -1, -1):
        if '\\begin{table' in lines[i]:
            begin_idx = i
            break
            
    # Find \end{table... forwards
    end_idx = -1
    for i in range(label_idx, len(lines)):
        if '\\end{table' in lines[i]:
            end_idx = i
            break
            
    if begin_idx == -1 or end_idx == -1: return
    
    if to_star:
        lines[begin_idx] = lines[begin_idx].replace('\\begin{table}', '\\begin{table*}')
        lines[end_idx] = lines[end_idx].replace('\\end{table}', '\\end{table*}')
        
    # Find \centering
    centering_idx = -1
    for i in range(begin_idx, end_idx):
        if '\\centering' in lines[i]:
            centering_idx = i
            break
            
    if centering_idx != -1:
        inserts = []
        if stretch:
            inserts.append(f'\\renewcommand{{\\arraystretch}}{{{stretch}}}\n')
        if colsep:
            inserts.append(f'\\setlength{{\\tabcolsep}}{{{colsep}}}\n')
            
        for idx, insert_str in enumerate(inserts):
            lines.insert(centering_idx + 1 + idx, insert_str)
            
    # Wrap tabular in resizebox if requested
    if wrap_resize:
        # Find \begin{tabular}
        tabular_begin_idx = -1
        # Need to re-find end_idx because we might have inserted lines
        end_idx = -1
        for i in range(label_idx, len(lines)):
            if '\\end{table' in lines[i]:
                end_idx = i
                break
        
        for i in range(begin_idx, end_idx):
            if '\\begin{tabular}' in lines[i] or '\\begin{tabular}{' in lines[i]:
                tabular_begin_idx = i
                break
                
        tabular_end_idx = -1
        for i in range(tabular_begin_idx, end_idx):
            if '\\end{tabular}' in lines[i]:
                tabular_end_idx = i
                break
                
        if tabular_begin_idx != -1 and tabular_end_idx != -1:
            lines.insert(tabular_end_idx + 1, '}\n')
            lines.insert(tabular_begin_idx, '\\resizebox{\\textwidth}{!}{%\n')

# Table 1: label tab:datasets
process_table('\\label{tab:datasets}', to_star=True, stretch='1.3', colsep='8pt', wrap_resize=False)

# Table 2: label tab:loss_benchmark
process_table('\\label{tab:loss_benchmark}', to_star=False, stretch='1.35', colsep='6pt', wrap_resize=False)

# Table 3: label tab:tree_comparison
process_table('\\label{tab:tree_comparison}', to_star=True, stretch='1.3', colsep='7pt', wrap_resize=False)

# Table 4: label tab:per_dataset
process_table('\\label{tab:per_dataset}', to_star=True, stretch='1.3', wrap_resize=False)

# Table 5: label tab:significance
process_table('\\label{tab:significance}', to_star=True, stretch='1.3', wrap_resize=True)

# Table 6: label tab:telemetry
process_table('\\label{tab:telemetry}', to_star=True, stretch='1.3', wrap_resize=True)

# Table 7: label tab:optimizer
process_table('\\label{tab:optimizer}', to_star=True, stretch='1.3', wrap_resize=True)

# Table 8: label tab:arch
process_table('\\label{tab:arch}', to_star=True, stretch='1.3', wrap_resize=True)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
