import copy
import datetime
from enum import IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np

# データの項目
class acc(IntEnum):
    time = 0
    acc_x = auto()
    acc_y = auto()
    acc_z = auto()
    gyro_x = auto()
    gyro_y = auto()
    gyro_z = auto()
    angle_x = auto()
    angle_y = auto()
    angle_z = auto()
    step = auto()


class rtk(IntEnum):
    date = 0
    time = auto()
    latitude = auto()
    longitude = auto()


def deg2rad(x):
    return x * np.pi/180


def rad2deg(x):
    return x * 180/np.pi


def calcDistance(latitude1, longitude1, latitude2, longitude2):
    GRS80_A = 6378137.000  # 長半径 a(m)
    GRS80_E2 = 0.00669438002301188  # 第一遠心率  eの2乗
    R = 6378137  # 赤道半径[m]

    my = deg2rad((latitude1 + latitude2) / 2.0)

    # 卯酉線曲率半径を求める(東と西を結ぶ線の半径)
    sinMy = np.sin(my)
    w = np.sqrt(1.0 - GRS80_E2 * sinMy * sinMy)
    n = GRS80_A / w

    # 子午線曲線半径を求める(北と南を結ぶ線の半径)
    mnum = GRS80_A * (1 - GRS80_E2)
    m = mnum / (w * w * w)

    deltaLatitude = deg2rad(latitude2 - latitude1)
    deltaLongitude = deg2rad(longitude2 - longitude1)

    deltaX = n * np.cos(my) * deltaLongitude
    deltaY = m * deltaLatitude
    distance = (deltaX ** 2 + deltaY ** 2) ** 0.5
    angle = np.arctan2(deltaY, deltaX)
    return distance, angle


def timeSub(time1, time2):
    return (time1 - time2).total_seconds()


def loadAccData(filename):
    accdata = np.loadtxt(filename, delimiter=',', skiprows=1,
                         unpack=True, dtype=str)

    acctime = []
    for i in range(len(accdata[acc.time])):
        acctime.append(datetime.datetime.strptime(
            accdata[acc.time][i], '%Y/%m/%d %H:%M:%S.%f'))
    accdict = {}
    accdict['time'] = acctime

    accdict['acc_x'] = list(accdata[acc.acc_x].astype('f8'))
    accdict['acc_y'] = list(accdata[acc.acc_y].astype('f8'))
    accdict['acc_z'] = list(accdata[acc.acc_z].astype('f8'))

    accdict['gyro_x'] = list(accdata[acc.gyro_x].astype('f8'))
    accdict['gyro_y'] = list(accdata[acc.gyro_y].astype('f8'))
    accdict['gyro_z'] = list(accdata[acc.gyro_z].astype('f8'))

    accdict['mag_x'] = list(accdata[acc.angle_x].astype('f8'))
    accdict['mag_y'] = list(accdata[acc.angle_y].astype('f8'))
    accdict['mag_z'] = list(accdata[acc.angle_z].astype('f8'))

    accdict['step'] = list(accdata[acc.step].astype('f8'))
    accdict['stepflag'] = [0]
    for i in range(1, len(accdict['step'])):
        accdict['stepflag'].append(accdict['step'][i] - accdict['step'][i-1])

    return accdict


def loadRTKData(filename):
    rtkdata = np.loadtxt(filename, delimiter=',', skiprows=1,
                         unpack=True, dtype=str)
    latitude = rtkdata[rtk.latitude].astype('f8')
    longitude = rtkdata[rtk.longitude].astype('f8')
    rtkdict = {}
    rtkdict['time'] = []
    rtkdict['speed'] = []
    rtkdict['angle'] = []
    for i in range(len(rtkdata[rtk.date])):
        # 時刻
        time = rtkdata[rtk.date][i] + ' ' + rtkdata[rtk.time][i]
        time = datetime.datetime.strptime(time, '%Y/%m/%d %H:%M:%S.%f')
        time += datetime.timedelta(hours=9)
        time -= datetime.timedelta(seconds=18)
        if i == 0:
            starttime = time
            continue
        rtkdict['time'].append(time)

        # 速度算出
        if i == 1:
            elapsedTime = timeSub(rtkdict['time'][i-1], starttime)
        else:
            elapsedTime = timeSub(rtkdict['time'][i-1], rtkdict['time'][i-2])
        distance, angle = calcDistance(latitude[i-1], longitude[i-1],
                                       latitude[i],   longitude[i]) 
        speed = distance / elapsedTime
        rtkdict['speed'].append(speed)
        rtkdict['angle'].append(angle)
    return rtkdict


# 外れ値削除
def outlierRemoval(time, data, threshold):
    i = len(time) - 1
    while i >= 0:
        if data[i] > threshold:
            del time[i]
            del data[i]
        i -= 1


# 線形補完
def linearInterpolation(time, data, cycle):
    i = len(time) - 1
    while i >= 0:
        timediff = timeSub(time[i], time[i-1])
        if timediff > cycle:
            nowtime = time[i] - datetime.timedelta(seconds=cycle)
            time.insert(i, nowtime)
            speeddiff = data[i] - data[i-1]
            nowacc = speeddiff / timediff
            nowspeed = data[i] - nowacc * (timediff-0.2)
            data.insert(i, nowspeed)
        else:
            i -= 1


# 5[Hz] から 1[Hz] に
def change1Hz(time, data, freq):
    time_1Hz = []
    data_1Hz = []
    for i in range(int(len(time) / freq)):
        time_1Hz.append(time[freq*(i+1)-1])
        data_1Hz.append(np.average(data[freq*i:freq*(i+1)]))
    return time_1Hz, data_1Hz


# ローパスフィルタ（移動平均）
def LPF(input, num=5):
    coefficient = np.ones(num) / num
    output = np.convolve(input, coefficient, mode='same')  # 移動平均
    return output


# filename = '0912_1800'
# filename = '0912_1815'
# filename = '0925'
# filename = '1010'
# filename = '1104_1607'
# filename = '1104_1650'
# filename = '1104_1744'
# filename = '1106_1059'
# filename = '1106_1120'
# filename = '1106_1136'
# filename = '1106_1152'
# filename = '1120_1519'
# filename = '1121_1625'
# filename = '1204_1956'
# filename = '1204_2012'
filename = '1211_1649'

# データの読み込み
accdata = loadAccData('data/' + filename + '/acc.csv')
rtkdata = loadRTKData('data/' + filename + '/rtk.csv')
speed = {}
speed['time'] = copy.copy(rtkdata['time'])
speed['speed'] = copy.copy(rtkdata['speed'])

# データ整形
outlierRemoval(speed['time'], speed['speed'], threshold=3.0)
linearInterpolation(speed['time'], speed['speed'], cycle=0.2)
speed_1Hz = {}
speed_1Hz['time'], speed_1Hz['speed'] = \
    change1Hz(speed['time'], speed['speed'], freq=5)

# 加速度データの不要な部分削除
starttime = speed_1Hz['time'][0] - datetime.timedelta(seconds=1)
i = len(accdata['time'])-1
while i >= 0:
    if accdata['time'][i] < starttime or \
       accdata['time'][i] > rtkdata['time'][-1]:
        for col in accdata.values():
            del col[i]
    i -= 1

print(accdata['time'][0])
print(speed_1Hz['time'][0])

# ファイル書き込み
datanum = 50
f = open('data/' + filename + '/ml.csv', mode='w')
f.write('time[s], speed[m/s], accwave_x({0})[G], accwave_y({0})[G],'
        'accwave_z({0})[G]\n'.format(datanum))
for i in range(20, len(speed_1Hz['speed'])):
    if datanum * (i+1) > len(accdata['time']):
        break
    f.write(str(i) + ', ' + str(speed_1Hz['speed'][i]) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_x'][datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_y'][datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_z'][datanum*i+j] / 9.8) + ', ')
    f.write('\n')
f.close

# datanum = 50
# f = open('data/ML_angle/' + filename + '.csv', mode='w')
# f.write('time[s], angle[deg], gyro_x({0})[G], mag_z({0})[G]\n'.format(datanum))
# for i in range(len(rtkdata['angle'])):
#     if datanum * (i+1) > len(accdata['time']):
#         break
#     f.write(str(i) + ', ' + str(rtkdata['angle'][i]) + ', ')
#     for j in range(datanum):
#         f.write(str(accdata['gyro_x'][10*i+j] / 9.8) + ', ')
#     for j in range(datanum):
#         f.write(str(accdata['mag_z'][10*i+j] / 9.8) + ', ')
#     f.write('\n')
# f.close

# # 時系列処理
# gyro = []
# gyro.append(accdata['gyro_x'])
# gyro.append(accdata['gyro_y'])
# gyro.append(accdata['gyro_z'])
# angleByGyro = []
# angleByGyro.append([0.0])
# angleByGyro.append([0.0])
# angleByGyro.append([0.0])
# Xlist = [0.0]
# Ylist = [0.0]
# for i in range(1, len(accdata['gyro_x'])):
#     for j in range(3):
#         angle = angleByGyro[j][i-1] - gyro[j][i]
#         if angle > 180:
#             angle -= 360
#         elif angle < -180:
#             angle += 360
#         angleByGyro[j].append(angle)

#     if accdata['stepflag'][i] > 0:
#         stride = 0.5
#         Xlist.append(Xlist[-1] + stride * np.cos(accdata['mag_z'][i]))
#         Ylist.append(Ylist[-1] + stride * np.sin(accdata['mag_z'][i]))

# # 微分
# def differential(time, data):
#     output = [0.0]
#     for i in range(1, len(time)):
#         timeDiff = timeSub(time[i], time[i-1])
#         angleDiff = data[i] - data[i-1]
#         if angleDiff > 180:
#             angleDiff -= 360
#         elif angleDiff < -180:
#             angleDiff += 360
#         output.append(angleDiff / timeDiff)
#     return output

# グラフ表示
fig = plt.figure(figsize=(8, 8))

# 加速度
graph1 = fig.add_subplot(211)
acc_x = list(map(lambda x: x / 9.8, accdata['acc_x']))
acc_y = list(map(lambda x: x / 9.8, accdata['acc_y']))
acc_z = list(map(lambda x: x / 9.8, accdata['acc_z']))
graph1.plot(accdata['time'], accdata['stepflag'], marker='.')
graph1.plot(accdata['time'], acc_x, marker='.', label='acc_x')
graph1.plot(accdata['time'], acc_y, marker='.', label='acc_y')
graph1.plot(accdata['time'], acc_z, marker='.', label='acc_z')
graph1.plot(speed['time'], speed['speed'], marker='.', label='speed')
graph1.plot(speed_1Hz['time'], speed_1Hz['speed'],
            marker='.', label='speed_1Hz')
graph1.grid()
graph1.legend()

# 角度
graph2 = fig.add_subplot(212)
rtkangle = list(map(rad2deg, rtkdata['angle']))
mag = list(map(rad2deg, accdata['mag_z']))
graph2.plot(rtkdata['time'], rtkangle, label='rtk')
graph2.plot(accdata['time'], mag, label='mag', alpha=0.7)
# graph2.plot(accdata['time'], LPF(angleByGyro[0], 10), label='gyro', alpha=0.7)
graph2.grid()
graph2.legend()

plt.show()
