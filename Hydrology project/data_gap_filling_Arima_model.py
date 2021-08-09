import numpy
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler




numpy.random.seed(7)
dataframe = read_csv('Data set/using data/monthly_data.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = dataset[:,1]
dataset = dataset[:,numpy.newaxis]

autocorrelation_plot(dataset)
pyplot.show()

train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

history = [x for x in train]
prediction = list()
for i in range(len(test)):
    model = ARIMA(history,order = (12,1,0))
    model = model.fit(disp = 0)
    output = model.forecast()
    prediction.append(output[0])
    history.append(test[i])
    print('predicted=%f, expected=%f' % (output[0], test[i]))

#error = mean_squared_error(test, predictions)
metrics.r2_score(test, prediction)
pyplot.plot(test)
pyplot.plot(prediction, color='red')
plt.title("test vs test predict")
plt.xlabel('timesteps')  
plt.ylabel('precipitation')
plt.legend(["Original","Predicted"], loc = "upper right")
pyplot.show()

MIndices = read_csv('Data set/using data/missing indices.csv', engine='python')
MIndices = MIndices.values
MIndices = MIndices.astype('float32')
indices_size = int(len(MIndices))
original_data_in_gap = numpy.empty([indices_size,1],dtype = 'float')
predicted_data_in_gap = numpy.empty([indices_size,1],dtype = 'float')
original_data_in_gap[:,:] = numpy.nan
predicted_data_in_gap[:,:] = numpy.nan

for i in range(indices_size):
    k = int(MIndices[i,0])
    original_data_in_gap[i] = test[k+12]
    predicted_data_in_gap[i,0] = prediction[k+12]

prediction = numpy.array(prediction)
    
plt.plot(original_data_in_gap)
plt.plot(predicted_data_in_gap)
plt.title("Data for Missing positions")
plt.xlabel('missing point indices in dataset')  
plt.ylabel('precipitation')
plt.legend(["Original","Predicted"], loc = "upper right")

metrics.r2_score(original_data_in_gap, predicted_data_in_gap)
Final_score_without_normalisation = math.sqrt(mean_squared_error(original_data_in_gap, predicted_data_in_gap))
print(Final_score_without_normalisation)

scaler = MinMaxScaler(feature_range=(0, 1))
original_data_in_gap_scaled = scaler.fit_transform (original_data_in_gap)
predicted_data_in_gap_scaled = scaler.fit_transform (predicted_data_in_gap)
Final_score_with_normalisation = math.sqrt(mean_squared_error(original_data_in_gap_scaled, predicted_data_in_gap_scaled))
print(Final_score_with_normalisation)

metrics.r2_score(test, prediction)
scaled_test = scaler.fit_transform (test)
scaled_pred = scaler.fit_transform (prediction)
test_rmse_normalised = math.sqrt(mean_squared_error(scaled_test, scaled_pred))
print(test_rmse_normalised)










