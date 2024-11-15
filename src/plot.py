import matplotlib.pyplot as plt

def plot_rewards(data, avg):
    plt.figure(1)
    plt.title('Inference...')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.plot(data)
    plt.plot([avg] * len(data))  
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

    plot_rewards(rewards, avg)
    pass

if __name__ == '__main__':
    main()