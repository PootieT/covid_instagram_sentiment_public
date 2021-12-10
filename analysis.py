from scipy.stats import chi2_contingency
from scipy.stats import chi2
import pandas as pd
import matplotlib.pyplot as plt


def chi_square_test(table):
    """
    cite: https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
    :param table:
    :return:
    """
    stat, p, dof, expected = chi2_contingency(table)
    print('dof=%d' % dof)
    print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')


def plot_emotion_prediction():
    df = pd.read_csv("data/instagram_faces/Final_Emotion_Prediction.csv")
    text_emo_counts = df["Post_Predictions"].value_counts().to_dict()
    face_emo_counts = df["Emotion"].value_counts().to_dict()
    face_emo_counts = {k.lower(): v for k, v in face_emo_counts.items()}
    count_df = pd.DataFrame([text_emo_counts, face_emo_counts], index=["text", "face"]).fillna(0)
    ax = count_df.T.plot.bar()
    plt.title("Emotion Counts Predicted by Text and Face")
    plt.xlabel("Emotions")
    plt.ylabel("Counts")
    plt.xticks(rotation="horizontal")

    plt.show()


def get_instagram_emotion_vs_asian():
    df = pd.read_excel("data/instagram/Labeled_instagram_posts_related_to_covid.xlsx")
    df_pred = pd.read_csv("data/instagram/insta_bertweet.csv", index_col="index")
    df_pred = df_pred.reindex(list(range(df_pred.index.min(), df_pred.index.max() + 1)), fill_value=0)
    df["emotion"] = df_pred[["fear", "sadness", "joy", "anger"]].idxmax(axis="columns")
    a_col = 'Q5A.  If yes to Q5, what type of Asian'
    df.loc[df[a_col]!=1, a_col] = "No E. Asian Present"
    df.loc[df[a_col] == 1, a_col] = "E. Asian Present"
    emo_per_lang = df.groupby(['emotion', a_col]).size().unstack(fill_value=0)
    emo_per_lang.T.plot.bar()
    plt.title("Text Emotion vs E. Asian People Presence")
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.xticks(rotation="horizontal")
    pass


def get_instagram_emotion_vs_language():
    df = pd.read_csv("data/instagram/insta_predictions_prob.csv")
    emo_per_lang = df.groupby(['preds', '0']).size().unstack(fill_value=0).T
    emo_per_lang["sum"] = emo_per_lang["fear"]+emo_per_lang["sadness"]+emo_per_lang["anger"]+emo_per_lang["joy"]
    emo_per_lang = emo_per_lang.sort_values("sum", ascending=False)
    emo_per_lang = emo_per_lang.loc[emo_per_lang.index[:5]]
    emo_per_lang = emo_per_lang.drop(columns=["sum"])
    emo_per_lang.plot.bar()
    plt.xlabel("Top 10 languages")
    plt.ylabel("Emotion Count")
    plt.title("Emotion Label Per Language")
    plt.show()
    pass

if __name__ == "__main__":
    # img_pred_vs_rasian = [[97,  68, 24, 2],  # east asian
    #                       [115, 60, 28, 1]] # not east asian
    # chi_square_test(img_pred_vs_rasian)
    # both_pred_vs_rasian = [[131, 53, 6, 1],  # not east asian
    #                        [98,  23, 3, 2]]  # east asian
    # chi_square_test(both_pred_vs_rasian)
    # text_pred_vs_rasian_small = [[43, 6, 142],  # not east asian
    #                        [19, 2, 105]]  # east asian
    # chi_square_test(text_pred_vs_rasian_small)
    # text_pred_vs_asian = [[5616, 477, 163, 2759],  # not east asian
    #                       [236, 60, 11, 326]]  # east asian
    # chi_square_test(text_pred_vs_asian)
    # plot_emotion_prediction()
    # get_instagram_emotion_vs_asian()
    get_instagram_emotion_vs_language()