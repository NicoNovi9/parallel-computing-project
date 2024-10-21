import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test_results.csv')

plt.figure(figsize=(10, 6))

for depth in data['Depth'].unique():
    subset = data[data['Depth'] == depth]
    label = 'Sequential' if depth == 0 else f'Thread {depth}'
    plt.plot(subset['Size'], subset['Time (seconds)'], marker='o', label=label)

plt.xscale('log')
plt.yscale('log') 
plt.xlabel('Size')
plt.ylabel('Time (seconds)')
plt.title('Testing Merge Sort Performance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('merge_sort_performance.png') 
plt.close() 

