import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from helper.functions import slugify


def convert_carmel_excelfiles_to_csv():
    # dir = "/Volumes/untitled/seedgerm/Seed Germ Pictures/JIC/brassica-ASSYST/" #type: str
    #dir = "/Volumes/untitled/seedgerm/Seed Germ Pictures/4ImageStacks/"
    #dir = "/Volumes/untitled/seedgerm/Seed Germ Pictures/New_ImageStacks_Ji_softwareVersion2/"
    dir = "/Volumes/untitled/Syngenta/New_ImageStacks_Ji_softwareVersion2/"


    root, dirs, files = next(os.walk(dir))



    for dir in dirs:
        file_system(root+dir+"/")

    tomato_man_array = []
    tomato_auto_array = []

    for sub_dir in dirs:
        #calculate_accuracy(sub_dir, truth_path=root+sub_dir, output_path=root+sub_dir)
        calculate_accuracy(sub_dir, truth_path=root + sub_dir + "/count", output_path=root + "output/",
                           tomato_man_array=tomato_man_array, tomato_auto_array=tomato_auto_array)

    # tomato_man_array = np.array(tomato_man_array)
    # tomato_auto_array = np.array(tomato_auto_array)

    print(tomato_man_array)
    print(tomato_auto_array)

    t = [("T25", 0.25), ("T50", 0.5), ("T75", 0.75), ("T100", 1.0)]
    for (time_name, index_slice) in t:

        # concat across all the experiments and panels.
        for exp_man, exp_auto in zip(tomato_man_array, tomato_auto_array):
            # we sample on the second axis, because the first is the panel.
            man_t_vals = exp_man[:, :int(index_slice * exp_man.shape[1])]
            auto_t_vals = exp_auto[:, :int(index_slice * exp_auto.shape[1])]

            # sample percentage from the panels, then concat into the long format.
            man_t_vals = np.concatenate(man_t_vals)
            auto_t_vals = np.concatenate(auto_t_vals)

        r2 = r2_score(man_t_vals, auto_t_vals)
        print("ALL_tomatoe" + " " + time_name + " " + str(r2))

        plt.figure(figsize=(10, 10))
        plt.title("Pairwise scatter plot per seed at " + time_name + " manual vs. automatic \nR2 = " + str(r2))
        plt.scatter(man_t_vals, auto_t_vals, s=24)
        plt.xlabel("Manual scoring")
        plt.ylabel("Automatic scoring")
        # plt.show()
        plt.savefig(root + "output/All_Tomatoe_" + time_name + "_pairwise.png")
        plt.close()


def file_system(dir):
    root, dirs, files = next(os.walk(dir))
    for file in files:
        if file.startswith("."):
            continue

        if ".xlsx" in file:
            df = pd.read_excel(root + file)
            df.to_csv(root + file.split(".")[0] + ".csv", index=False)


def calculate_accuracy(name, truth_path=None, output_path=None, seed_n=50, tomato_man_array=None,
                       tomato_auto_array=None):

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #gather results
    data = "data/experiments/"+slugify(name)+"/results/panel_germinated_cumulative.csv"
    print(data)
    try:
        with open(data, 'r') as csvfile1, open(truth_path+".csv", 'r') as csvfile2:#, open(output_path+"_pairs.csv", 'w+') as csvfile3:
            preds = csv.reader(csvfile1, delimiter=',')
            truths = csv.reader(csvfile2, delimiter=',')

            #read the headers
            next(preds)
            next(truths)

            # do it twice if it is weird syngenta data
            # next(truths)



            preds = list(preds)
            truths = list(truths)

            # columns = [0] + list(range(1, len(truths[0]), 3))
            # print(columns)
            # from operator import itemgetter
            # truths = list(map(list, map(itemgetter(*columns), truths)))
            #print(truths)

            panel_manual = {}
            panel_automatic = {}

            #csvfile3.write("%s,%s,%s\n" % ('image_no', 'automatic', 'manual'))

            #dont want last row which has total seeds.
            for index, row in enumerate(preds[:-1]):
                image_no = int(row[0])
                #check if its truth is available.
                #loop until we hit the image no.

                found = None
                #GET THE MATCHING ROW.

                for tru in truths:
                    if int(tru[0]) == image_no:
                        found = tru

                if found is None:
                    continue

                for ind, (pan, tru) in enumerate(zip(row[1:], found[1:])):
                    if tru is None or tru is "":
                        continue

                    pan = float(pan)
                    tru = float(tru)
                    if panel_manual.get(ind) is None:
                        panel_manual[ind] = [tru]
                    else:
                        panel_manual[ind].append(tru)

                    if panel_automatic.get(ind) is None:
                        panel_automatic[ind] = [pan]
                    else:
                        panel_automatic[ind].append(pan)

                    #csvfile3.write("%d,%s,%s\n" % (image_no, pan, tru))

            man_vals = []
            auto_vals = []

            for key1, key2 in zip(panel_automatic,panel_manual):
                man_vals.append(panel_automatic[key1])
                auto_vals.append(panel_manual[key2])

            panel_r2 = {}
            for key1, key2 in zip(panel_automatic.keys(),panel_manual.keys()):
                panel_r2[key1] = r2_score(panel_automatic[key1], panel_manual[key2])

            avg_man = np.mean(np.array(panel_manual.values()), axis=0)
            if avg_man.shape[0] > 1:
                avg_man = [np.mean(avg_man[:i + 1]) for i in range(avg_man.shape[0] - 1)]
            avg_auto = np.mean(np.array(panel_automatic.values()), axis=0)
            if avg_auto.shape[0] > 1:
                avg_auto = [np.mean(avg_auto[:i + 1]) for i in range(avg_auto.shape[0] - 1)]

            # r2 = r2_score(avg_man, avg_auto)

            # plt.figure(figsize=(10, 10))
            # plt.title("Cumulative mean of the average across all panels for both manual and automatic \nR2 = " + str(r2))
            # plt.plot(avg_man, range(0, len(avg_man)), label="manual")
            # plt.plot(avg_auto, range(0, len(avg_auto)), label="automatic")
            # plt.legend()
            # plt.savefig(output_path + name+ "_average_line.png")
            # plt.close()

            man_vals = np.array(man_vals)
            auto_vals = np.array(auto_vals)

            if (name.__contains__("omato")):
                tomato_man_array.append(man_vals)
                tomato_auto_array.append(auto_vals)

            print(man_vals.shape)

            print(len(preds))
            t = [("T25", 0.25), ("T50", 0.5), ("T75", 0.75), ("T100", 1.0)]
            for (time_name, index_slice) in t:
                # we sample on the second axis, because the first is the panel.
                man_t_vals = man_vals[:, :int(index_slice * man_vals.shape[1])]
                auto_t_vals = auto_vals[:, :int(index_slice * auto_vals.shape[1])]

                # sample percentage from the panels, then concat into the long format.
                man_t_vals = np.concatenate(man_t_vals)
                auto_t_vals = np.concatenate(auto_t_vals)

                r2 = r2_score(man_t_vals, auto_t_vals)
                print(name + " " + time_name + " " + str(r2))

                plt.figure(figsize=(10, 10))
                plt.title("Pairwise scatter plot per seed at " + time_name + " manual vs. automatic \nR2 = " + str(r2))
                plt.scatter(man_t_vals, auto_t_vals, s=24)
                plt.xlabel("Manual scoring")
                plt.ylabel("Automatic scoring")
                # plt.show()
                plt.savefig(output_path + name + "_" + time_name + "_pairwise.png")
                plt.close()




    except IOError as e:
        print(e)
        print("no such file")

convert_carmel_excelfiles_to_csv()