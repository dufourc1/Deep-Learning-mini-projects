import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
# GPU nvdia GTX970m

df = pd.read_csv('results/results_gpu.csv')

df['ActivFunc'] = 0
df.loc[range(12),['ActivFunc']] = 'ReLU'
df.loc[range(12,19),['ActivFunc']] = 'Tanh'
df.loc[range(19,len(df)),['ActivFunc']] = 'LeakyReLU'

fig = plt.figure(figsize=(10,5))
# plt.title('Model accuracy on test set')
plt.grid(axis='y')

model_names = df.ModelName[:12]

# width of the bars
barWidth = 0.3

# Accuracy bars
bars_relu = df.meanAccuracy_te[df['ActivFunc'] == 'ReLU']
bars_tanh = df.meanAccuracy_te[df['ActivFunc'] == 'Tanh']
bars_Lrelu = df.meanAccuracy_te[df['ActivFunc'] == 'LeakyReLU']

x_relu = pd.array(range(12), dtype='float')
x_Lrelu = x_relu + 2*barWidth
x_tanh = x_relu + barWidth

x_relu[-5:] += barWidth

# Height of the error bars
yer_relu = df.stdAccuracy_te[df['ActivFunc'] == 'ReLU']
yer_tanh = df.stdAccuracy_te[df['ActivFunc'] == 'Tanh']
yer_Lrelu = df.stdAccuracy_te[df['ActivFunc'] == 'LeakyReLU']

transparent = (0, 0, 0, 0)
plt.bar(x_relu, bars_relu, width= barWidth, yerr=yer_relu, capsize=3, label='ReLU')
plt.bar(x_Lrelu[:len(bars_Lrelu)], bars_Lrelu, width = barWidth, yerr=yer_Lrelu, capsize=3, label='LeakyReLU')
plt.bar(x_tanh[:len(bars_tanh)], bars_tanh, width = barWidth, yerr=yer_tanh, capsize=3, label='Tanh')

# general layout
plt.ylim(0.4,1)
plt.xticks([r + barWidth for r in range(len(bars_relu))], model_names, rotation=25, ha = 'right')
plt.ylabel('Accuracy')
# left, right = plt.xlim()
# plt.xlim(left, right)
# plt.hlines(CV_per_class_results['Logistic Regression'].mean(), left, right, color='seagreen', linestyles='dashed', label='Best per-class accuracy')
# plt.hlines(CV_results['Logistic Regression'].mean(), left, right, color='steelblue', linestyles='dotted', label='Best overall accuracy')
plt.legend()


plt.savefig('results/plots/BarPlotResults-gpu.png', bbox_inches='tight', pad_inches=0.2)

################################################################################

fig = plt.figure(figsize=(10,5))
# plt.title('Training Time (seconds)')
plt.grid(axis='y')

model_names = df.ModelName[:12]

# width of the bars
barWidth = 0.5

# Time bars
bars_relu = df.drop(index=range(7))[df.drop(index=range(7))['ActivFunc'] == 'ReLU']['meanTime_tr']

x_relu = pd.array(range(len(bars_relu)), dtype='float')

# Height of the error bars
yer_relu = df.drop(index=range(7))[df.drop(index=range(7))['ActivFunc'] == 'ReLU']['stdTime_tr']

transparent = (0, 0, 0, 0)
plt.bar(x_relu, bars_relu, width= barWidth, yerr=yer_relu, capsize=3, label='ReLU')

# general layout
bpttom, top = plt.ylim()
plt.yscale('log')
plt.ylim(0.75,top)
plt.xticks([r for r in range(len(bars_relu))], model_names[7:], rotation=25, ha = 'right')
plt.ylabel('Time')
# plt.xlim(left, right)
# plt.hlines(CV_per_class_results['Logistic Regression'].mean(), left, right, color='seagreen', linestyles='dashed', label='Best per-class accuracy')
# plt.hlines(CV_results['Logistic Regression'].mean(), left, right, color='steelblue', linestyles='dotted', label='Best overall accuracy')
plt.legend()


plt.savefig('results/plots/BarPlotTime-slow-gpu.png', bbox_inches='tight', pad_inches=0.2)

################################################################################

fig = plt.figure(figsize=(10,5))
# plt.title('Model accuracy on test set')
plt.grid(axis='y')

model_names = df.ModelName[:12]

# width of the bars
barWidth = 0.3

# Accuracy bars
bars_relu = df.meanTime_tr[df['ActivFunc'] == 'ReLU']
bars_tanh = df.meanTime_tr[df['ActivFunc'] == 'Tanh']
bars_Lrelu = df.meanTime_tr[df['ActivFunc'] == 'LeakyReLU']

x_relu = pd.array(range(12), dtype='float')
x_Lrelu = x_relu + 2*barWidth
x_tanh = x_relu + barWidth

x_relu[-5:] += barWidth

# Height of the error bars
yer_relu = df.stdTime_tr[df['ActivFunc'] == 'ReLU']
yer_tanh = df.stdTime_tr[df['ActivFunc'] == 'Tanh']
yer_Lrelu = df.stdTime_tr[df['ActivFunc'] == 'LeakyReLU']

transparent = (0, 0, 0, 0)
plt.bar(x_relu, bars_relu, width= barWidth, yerr=yer_relu, capsize=3, label='ReLU')
plt.bar(x_Lrelu[:len(bars_Lrelu)], bars_Lrelu, width = barWidth, yerr=yer_Lrelu, capsize=3, label='LeakyReLU')
plt.bar(x_tanh[:len(bars_tanh)], bars_tanh, width = barWidth, yerr=yer_tanh, capsize=3, label='Tanh')

# general layout
plt.yscale('log')
# plt.ylim(0.4,1)
plt.xticks([r + barWidth for r in range(len(bars_relu))], model_names, rotation=25, ha = 'right')
plt.ylabel('Training time (sec)')
# left, right = plt.xlim()
# plt.xlim(left, right)
# plt.hlines(CV_per_class_results['Logistic Regression'].mean(), left, right, color='seagreen', linestyles='dashed', label='Best per-class accuracy')
# plt.hlines(CV_results['Logistic Regression'].mean(), left, right, color='steelblue', linestyles='dotted', label='Best overall accuracy')
plt.legend()


plt.savefig('results/plots/BarPlotTime-all-gpu.png', bbox_inches='tight', pad_inches=0.2)

################################################################################
# CPU on virtual machine results

df2 = pd.read_csv('results/results_cpu.csv')

df2['ActivFunc'] = 0
df2.loc[range(7),['ActivFunc']] = 'ReLU'
df2.loc[range(7,14),['ActivFunc']] = 'Tanh'
df2.loc[range(14,len(df2)),['ActivFunc']] = 'LeakyReLU'

df2.loc[df2.ModelName.eq('DropoutFullyConnectedBatchNorm'), 'stdTime_tr'] = 0

fig = plt.figure(figsize=(10,5))
# plt.title('Model accuracy on test set')
plt.grid(axis='y')

model_names = df2.ModelName[:7]

# width of the bars
barWidth = 0.3

# Accuracy bars
bars_relu = df2.meanAccuracy_te[df2['ActivFunc'] == 'ReLU']
bars_tanh = df2.meanAccuracy_te[df2['ActivFunc'] == 'Tanh']
bars_Lrelu = df2.meanAccuracy_te[df2['ActivFunc'] == 'LeakyReLU']

x_relu = pd.array(range(7), dtype='float')
x_Lrelu = x_relu + 2*barWidth
x_tanh = x_relu + barWidth

# Height of the error bars
yer_relu = df2.stdAccuracy_te[df2['ActivFunc'] == 'ReLU']
yer_tanh = df2.stdAccuracy_te[df2['ActivFunc'] == 'Tanh']
yer_Lrelu = df2.stdAccuracy_te[df2['ActivFunc'] == 'LeakyReLU']

transparent = (0, 0, 0, 0)
plt.bar(x_relu, bars_relu, width= barWidth, yerr=yer_relu, capsize=3, label='ReLU')
plt.bar(x_Lrelu[:len(bars_Lrelu)], bars_Lrelu, width = barWidth, yerr=yer_Lrelu, capsize=3, label='LeakyReLU')
plt.bar(x_tanh[:len(bars_tanh)], bars_tanh, width = barWidth, yerr=yer_tanh, capsize=3, label='Tanh')

# general layout
plt.ylim(0.4,1)
plt.xticks([r + barWidth for r in range(len(bars_relu))], model_names, rotation=25, ha = 'right')
plt.ylabel('Accuracy')
# left, right = plt.xlim()
# plt.xlim(left, right)
# plt.hlines(CV_per_class_results['Logistic Regression'].mean(), left, right, color='seagreen', linestyles='dashed', label='Best per-class accuracy')
# plt.hlines(CV_results['Logistic Regression'].mean(), left, right, color='steelblue', linestyles='dotted', label='Best overall accuracy')
plt.legend()


plt.savefig('results/plots/BarPlotResults-cpu.png', bbox_inches='tight', pad_inches=0.2)

################################################################################

fig = plt.figure(figsize=(10,5))
# plt.title('Model accuracy on test set')
plt.grid(axis='y')

model_names = df2.ModelName[:7]

# width of the bars
barWidth = 0.3

# Time bars
bars_relu = df2.meanTime_tr[df2['ActivFunc'] == 'ReLU']
bars_tanh = df2.meanTime_tr[df2['ActivFunc'] == 'Tanh']
bars_Lrelu = df2.meanTime_tr[df2['ActivFunc'] == 'LeakyReLU']

x_relu = pd.array(range(7), dtype='float')
x_Lrelu = x_relu + 2*barWidth
x_tanh = x_relu + barWidth

# Height of the error bars
yer_relu = df2.stdTime_tr[df2['ActivFunc'] == 'ReLU']
yer_tanh = df2.stdTime_tr[df2['ActivFunc'] == 'Tanh']
yer_Lrelu = df2.stdTime_tr[df2['ActivFunc'] == 'LeakyReLU']

transparent = (0, 0, 0, 0)
plt.bar(x_relu, bars_relu, width= barWidth, yerr=yer_relu, capsize=3, label='ReLU')
plt.bar(x_Lrelu[:len(bars_Lrelu)], bars_Lrelu, width = barWidth, yerr=yer_Lrelu, capsize=3, label='LeakyReLU')
plt.bar(x_tanh[:len(bars_tanh)], bars_tanh, width = barWidth, yerr=yer_tanh, capsize=3, label='Tanh')

# general layout
plt.ylim()
plt.xticks([r + barWidth for r in range(len(bars_relu))], model_names, rotation=25, ha = 'right')
plt.ylabel('Time')
# left, right = plt.xlim()
# plt.xlim(left, right)
# plt.hlines(CV_per_class_results['Logistic Regression'].mean(), left, right, color='seagreen', linestyles='dashed', label='Best per-class accuracy')
# plt.hlines(CV_results['Logistic Regression'].mean(), left, right, color='steelblue', linestyles='dotted', label='Best overall accuracy')
plt.legend()


plt.savefig('results/plots/BarPlotTimes-cpu.png', bbox_inches='tight', pad_inches=0.2)
