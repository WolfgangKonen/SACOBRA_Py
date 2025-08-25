import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
fmri = sns.load_dataset("fmri")
# fmri: data frame with columns subject,timepoint,signal,region,event

x = fmri.groupby(["event", "region", "timepoint"]).std()

sns.lineplot(x="timepoint", y="signal", errorbar=("ci",95), hue="region",
             style="event", palette="inferno", data=fmri)
plt.show()
dummy=0