import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(data, avg):
    plt.figure(1)
    plt.title('Inference...')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.plot(data)
    plt.plot([avg] * len(data))  
    plt.show()

def plot_gravity(data, above_200):
    plt.figure(2)
    plt.title('Percentage of Trials with reward >= 200 for different gravities')
    plt.xlabel('Acceleration due to gravity')
    plt.ylabel('Percentage of Trials')
    # plt.grid(True)
    plt.bar(data.keys(), np.array(list(above_200.values()))/np.array(list(data.values()))*100, color='red', width=0.4)
    plt.show()


def main():
    rewards = []
    avg = 0.0
    with open('output/rewards.txt', 'r') as fd:
        lines = fd.readlines()
        for l in lines:
            rewards += [float(l.strip())]
            avg += rewards[-1]
    avg = avg/len(rewards)
    print('average:', avg)

    # plot_rewards(rewards, avg)

    gravities = {'1-2':0, '2-3':0, '3-4':0, '4-5':0, '5-6':0, '6-7':0, '7-8':0, '8-9':0, '9-10':0}
    above_200 = {'1-2':0, '2-3':0, '3-4':0, '4-5':0, '5-6':0, '6-7':0, '7-8':0, '8-9':0, '9-10':0}
    # gravities = {'6-7':0, '7-8':0, '8-9':0, '9-10':0}
    # above_200 = {'6-7':0, '7-8':0, '8-9':0, '9-10':0}
    with open('output/gravitys.txt', 'r') as fd:
        lines = fd.readlines()
        for idx, l in enumerate(lines):
            g = float(l.strip())

            if abs(g) >= 1.0 and abs(g) < 2.0:
                gravities['1-2'] += 1
                if rewards[idx] >= 200:
                    above_200['1-2'] += 1
            elif abs(g) >= 2.0 and abs(g) < 3.0:
                gravities['2-3'] += 1
                if rewards[idx] >= 200:
                    above_200['2-3'] += 1
            elif abs(g) >= 3.0 and abs(g) < 4.0:
                gravities['3-4'] += 1
                if rewards[idx] >= 200:
                    above_200['3-4'] += 1
            elif abs(g) >= 4.0 and abs(g) < 5.0:
                gravities['4-5'] += 1
                if rewards[idx] >= 200:
                    above_200['4-5'] += 1
            elif abs(g) >= 5.0 and abs(g) < 6.0:
                gravities['5-6'] += 1
                if rewards[idx] >= 200:
                    above_200['5-6'] += 1
            elif abs(g) >= 6.0 and abs(g) < 7.0:
                gravities['6-7'] += 1
                if rewards[idx] >= 200:
                    above_200['6-7'] += 1
            elif abs(g) >= 7.0 and abs(g) < 8.0:
                gravities['7-8'] += 1
                if rewards[idx] >= 200:
                    above_200['7-8'] += 1
            elif abs(g) >= 8.0 and abs(g) < 9.0:
                gravities['8-9'] += 1
                if rewards[idx] >= 200:
                    above_200['8-9'] += 1
            elif abs(g) >= 9.0 and abs(g) < 10.0:
                gravities['9-10'] += 1
                if rewards[idx] >= 200:
                    above_200['9-10'] += 1

    plot_gravity(gravities, above_200)
    
    pass

if __name__ == '__main__':
    main()