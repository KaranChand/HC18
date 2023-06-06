import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "DICE": {
        "trimester_1": {
            "max": 99.70283001124314,
            "min": 91.40084168391616,
            "std": 1.140526005336919,
            "25pc": 99.18643915816416,
            "50pc": 99.43295577055011,
            "75pc": 99.54585638222127,
            "mean": 99.1586813434653
        },
        "trimester_2": {
           "max": 99.85732926801899,
            "min": 95.15286976849099,
            "std": 0.46100655421659276,
            "25pc": 99.4859047128788,
            "50pc": 99.5770161473437,
            "75pc": 99.65325998333498,
            "mean": 99.48798369103754
        },
        "trimester_3": {
            "max": 99.84492629302878,
            "min": 96.88320532707804,
            "std": 0.4603054310770239,
            "25pc": 99.45649498833315,
            "50pc": 99.54621904714325,
            "75pc": 99.62272504832386,
            "mean": 99.4561379246103
        }
        },
    
    "absolute_difference": {
        "trimester_1": {
            "max": 6.16166162499141,
            "min": 0.0026408279932326195,
            "std": 0.8113065677922291,
            "25pc": 0.3068313336915267,
            "50pc": 0.4003269983640507,
            "75pc": 0.5448808423872009,
            "mean": 0.5311887122937907,
        },
        "trimester_2": {
             "max": 5.021277311139869,
            "min": 0.000240997831298273,
            "std": 0.5646659365129535,
            "25pc": 0.4478971913125065,
            "50pc": 0.6260147753899616,
            "75pc": 0.798205033284404,
            "mean": 0.6972638045103273,
        },
        "trimester_3": {
            "max": 7.554321878620556,
            "min": 0.06425642503097606,
            "std": 1.0917479855062031,
            "25pc": 0.6386118985220435,
            "50pc": 1.1750706459259277,
            "75pc": 1.454021769396462,
            "mean": 1.2001882077821304,
        }
    },
    "hausdorff_distance": {
        "trimester_1": {
            "max": 3.4023396253993576,
            "min": 0.07432923262057371,
            "std": 0.4504060731420179,
            "25pc": 0.1420524946581353,
            "50pc": 0.1833433843493421,
            "75pc": 0.250936398593844,
            "mean": 0.2825086132645166,
        },
        "trimester_2": {
            "max": 2.751661477972805,
            "min": 0.113191977143,
            "std": 0.3268106269054178,
            "25pc": 0.238637924194,
            "50pc": 0.27787668304912183,
            "75pc": 0.352264032807,
            "mean": 0.3637855738521668,
        },
        "trimester_3": {
            "max": 3.757570928223123,
            "min": 0.20840751338675412,
            "std": 0.5537751301169123,
            "25pc": 0.42913234807045236,
            "50pc": 0.536794002567,
            "75pc": 0.666951245209316,
            "mean": 0.6623335877872517,
        }
    },
    "difference": {
        "trimester_1": {
            "max": 6.16166162499141,
            "min": -1.1494257802494872,
            "std": 0.933788356560874,
            "25pc": -0.28172203404781726,
            "50pc": 0.306111452785359,
            "75pc": 0.47547692234254413,
            "mean": 0.26889649434283014,
        },
        "trimester_2": {
            "max": 1.7415181990976407,
            "min": -5.021277311139869,
            "std": 0.8934239519967865,
            "25pc": -0.4984560689954378,
            "50pc": 0.41683495931374637,
            "75pc": 0.696345855750991,
            "mean": 0.09420938688268495,
        },
        "trimester_3": {
            "max": 2.0205943355688873,
            "min": -7.554321878620556,
            "std": 1.6193847482964212,
            "25pc": -0.6386118985220435,
            "50pc": 0.49755552280629445,
            "75pc": 1.2993637502955835,
            "mean": 0.2009837371100841,
        }
    }
}

trimesters = ['trimester_1', 'trimester_2', 'trimester_3']
palette = sns.color_palette("Set2", len(trimesters))

fig, axes = plt.subplots(nrows=len(data.keys()), ncols=1, figsize=(10, 20), sharex=True)

for i, measure in enumerate(data.keys()):
    ax = axes[i]
    
    for j, trimester in enumerate(trimesters):
        mean = data[measure][trimester]["mean"]
        std = data[measure][trimester]["std"]
        min_val = data[measure][trimester]["min"]
        max_val = data[measure][trimester]["max"]
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)

        ax.plot(x, y, label=trimester, color=palette[j], linewidth=3)
        ax.axvline(min_val, color=palette[j], linestyle='-', linewidth=4, alpha=0.3)
        ax.axvline(max_val, color=palette[j], linestyle='-', linewidth=6, alpha=0.3)

    ax.set_ylabel('Probability Density', fontsize=15)
    ax.set_title(f'{measure}', fontsize=15)
    ax.grid(True)
    ax.legend(fontsize=15)

plt.xlabel('X', fontsize=15)
plt.xlim(left=-10, right=110)
plt.tight_layout()
plt.savefig("gaussians_nnUNet.pdf")
plt.show()
