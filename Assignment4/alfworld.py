import os
import sys
import yaml
import json
import openai
import alfworld
from openai import OpenAI
from alfworld.agents.environment import AlfredTWEnv

# Load OpenAI API client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define LLM function that uses OpenAI's ChatGPT
def llm(messages, **kwargs) -> str:
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        prompt=messages,
        **kwargs
    )
    return chat_completion.choices[0].text.strip()

# Initialize ALFWorld environment
def init_alfworld_env():
    os.environ['ALFWORLD_DATA'] = 'alfworld\alfworld\data'
    with open("base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)
    return env

# Load ICL examples
def load_icl_examples():
    with open("./alfworld_act.json", "r") as file:
        return json.load(file)

# Process observation
def process_obs(obs):
    if obs.startswith("You arrive at loc "):
        obs = obs[obs.find('. ') + 2:]
    return obs

# Implement Act framework
def Act(prompt, obs, env):
    action_history = []
    observation_history = [obs]
    prompt = 'Think: '
    for i in range(50):
        try:
            action = llm(prompt + '\n'.join(observation_history) + '\n>', stop=['\n']).strip()
            action_history.append(action)

            if action.startswith('Think:'):
                observation = 'OK.'
                observation_history.append(observation)
            else:
                observation, reward, done, info = env.step([action])
                observation, reward, done = process_obs(observation[0]), info['won'][0], done[0]
                observation_history.append(observation)

                if done:
                    return reward
            print(f"Action: {action}\nObservation: {observation}")
            
        except Exception as e:
            print(f"Error occurred during Act execution: {e}")
            return 0
    
    return 0

# Implement ReAct framework
def ReAct(env, icl_examples):
    reward = [0]*6
    cnt = [0]*6
    Task_list = ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place", "pick_cool_then_place", "pick_two_obj", "look_at_obj"]
    for _ in range(10):
        observation, info = env.reset()  # get a new task
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        for task in Task_list:
            if name.startswith(task):
                prompt = "You are an agent and you will complete a task.\nHere are some examples:\n" + icl_examples[f"{task}_0"] + icl_examples[f"{task}_1"] + icl_examples[f"{task}_2"] + f"Here is the task: {name}\n"
                print(prompt + '\n>')
                r = Act(prompt, observation[0], env)
                reward[Task_list.index(task)] += r
                cnt[Task_list.index(task)] += 1
                break
        print(f"Reward: {reward}, Count: {cnt}")
        print("---------------------\n")

# Main execution
if __name__ == "__main__":
    env = init_alfworld_env()
    icl_examples = load_icl_examples()
    ReAct(env, icl_examples)
