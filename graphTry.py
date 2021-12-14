import os

import matplotlib.pyplot as plt

FGnet_path = "./datasets/FGNET/images/"

images_path = os.listdir(FGnet_path)
finalData = []
for i, path in enumerate(images_path):
    names = path.split(".")[0]
    names = names.split("A")
    names[1] = names[1].replace("a", "")
    names[1] = names[1].replace("b", "")
    # finalData.append([int(names[0]), int(names[1])])
    finalData.append([names[0], int(names[1])])

# print(finalData)

collectingData = {}
# collectingData["sd"] = []

for data in finalData:
    try:
        collectingData[data[0]].append(data[1])
    except:
        # print("An exception occurred")
        collectingData[data[0]] = [data[1]]

# print(collectingData)

finalCollectionLabels = []
finalCollectionValues = []
imgCount = []
finalAvgDiff = []
for data in collectingData:
    sum = 0
    lastIndex = 0
    deffCount = 0
    deffSum = 0
    for i, ages in enumerate(collectingData[data]):
        try:
            if id((collectingData[data][i + 1] - ages) != 0):
                deffSum = deffSum + (collectingData[data][i + 1] - ages)
                deffCount = deffCount + 1
        except:
            pass

        sum = sum + ages
        lastIndex = i
    avgDef = deffSum / deffCount
    avg = sum / (lastIndex + 1)
    finalCollectionLabels.append(int(data))
    finalCollectionValues.append(avg)
    imgCount.append((lastIndex + 1))
    finalAvgDiff.append(avgDef)


# print(finalCollection[0])

sumImgCount = 0
totImgCount = 0
for var in imgCount:
    sumImgCount += var
    totImgCount += 1

avgImgCount = sumImgCount / totImgCount
print(avgImgCount)


###############################################
sumOffinalAvgDiff = 0
totFinalAvgDiff = 0
for var in finalAvgDiff:
    sumOffinalAvgDiff = sumOffinalAvgDiff + var
    totFinalAvgDiff = totFinalAvgDiff + 1

avgFinalAvgDiff = sumOffinalAvgDiff / totFinalAvgDiff
print(avgFinalAvgDiff)
plt.bar(
    finalCollectionLabels,
    imgCount,
    tick_label=finalCollectionLabels,
    width=0.4,
    color=["red", "green"],
)


# naming the x-axis
plt.xlabel("Labels")
# naming the y-axis
plt.ylabel("Number of imgs")
# plot title
plt.title("Number of images for eash person")

# function to show the plot
plt.show()

## should make avg age diffrence between images for each person
plt.bar(
    finalCollectionLabels,
    finalCollectionValues,
    tick_label=finalCollectionLabels,
    width=0.4,
    color=["red", "green"],
)


# naming the x-axis
plt.xlabel("Labels")
# naming the y-axis
plt.ylabel("Avg age")
# plot title
plt.title("Avarge Age")

# function to show the plot
plt.show()

#############################

plt.bar(
    finalCollectionLabels,
    finalAvgDiff,
    tick_label=finalCollectionLabels,
    width=0.4,
    color=["red", "green"],
)


# naming the x-axis
plt.xlabel("Labels")
# naming the y-axis
plt.ylabel("Avg age diff")
# plot title
plt.title("Avarge difference in Age between images for each Person")

# function to show the plot
plt.show()
