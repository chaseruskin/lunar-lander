# Entry script for the lunar lander project.

import gymnasium as gym
import PIL.Image
import os

def main():
    '''
    Create the environment, run trials, and record results.
    '''
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                render_mode='rgb_array')
    env.reset()
    # get the first frame
    img = PIL.Image.fromarray(env.render())
    # save the image to the file
    os.makedirs('output', exist_ok=True)
    img.save('output/frame0.png')
    pass


if __name__ == '__main__':
    main()