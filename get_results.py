import pandas as pd
import numpy as np

# DE SETUP VOOR DE DATAFRAME
header = np.array(['filename', 'center_x_mm', 'center_y_mm', 'semi_axes_a_mm', 'semi_axes_b_mm', 'angle_rad'])
content = np.zeros((335, 6), dtype=object)
content[:, 0] = '000_HC.png'
content[:, 1:] = 1.5
df = pd.DataFrame(content, columns=header)

print(df.head())

### 
# VUL HIER DE PANDAS DATAFRAME MET DE GOEDE WAARDES
###



###
###
###

# HIER WORDT HET GESAVED NAAR CSV
df.to_csv('test_results.csv', index=False)
