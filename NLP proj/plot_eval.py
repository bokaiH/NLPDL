'''
可视化评估结果
'''
import numpy as np 
import matplotlib.pyplot as plt
import  pandas as pd


def plot_bar(data):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    axes[0].bar(data["Method"], data["Similarity Score"], color=['blue', 'green', 'red'])
    axes[0].set_title('Similarity Scores')
    axes[0].set_ylabel('Score')
    for i, score in enumerate(data["Similarity Score"]):
        axes[0].text(i, score, f'{score:.2f}', ha='center', va='bottom')

    axes[1].bar(data["Method"], data["Paragraph Counts"], color=['purple', 'orange', 'cyan'])
    axes[1].set_title('Paragraph Counts')
    axes[1].set_ylabel('Count')
    for i, count in enumerate(data["Paragraph Counts"]):
        axes[1].text(i, count, str(count), ha='center', va='bottom')

    axes[2].bar(data["Method"], data["Entities Covered"], color=['yellow', 'pink', 'lightblue'])
    axes[2].set_title('Entities Covered')
    axes[2].set_ylabel('Number of Entities')
    for i, count in enumerate(data["Entities Covered"]):
        axes[2].text(i, count, str(count), ha='center', va='bottom')

    axes[3].bar(data["Method"], data["BLEU Score"], color=['green', 'orange', 'pink'])
    axes[3].set_title('Bleu Scores')
    axes[3].set_ylabel('Score')
    for i, count in enumerate(data["BLEU Score"]):
        axes[3].text(i, count, str(count), ha='center', va='bottom')

    axes[4].bar(data["Method"], data["METEOR Score"], color=['blue', 'cyan', 'yellow'])
    axes[4].set_title('Meteor Scores')
    axes[4].set_ylabel('Score')
    for i, count in enumerate(data["METEOR Score"]):
        axes[4].text(i, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_radar(data):

    df_updated = pd.DataFrame(data)

    df_norm = df_updated.copy()
    for col in df_updated.columns[1:]:
        max_value = df_updated[col].max()
        df_norm[col] = df_updated[col] / max_value

    labels=np.array(df_norm.columns[1:])
    stats=df_norm.loc[:, df_norm.columns[1:]].values

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    stats = np.concatenate((stats, stats[:,[0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(len(df_norm)):
        ax.plot(angles, stats[i], label=df_norm['Method'][i])
        ax.fill(angles, stats[i], alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title('Method Comparison using Radar Chart', size=20, color='blue', y=1.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.show()



