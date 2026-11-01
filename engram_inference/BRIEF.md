# STACK on the Engram datasets

This note describes a first pass of the published STACK model on the two single-cell datasets from the Engram meeting. Group A is the cortical organoid atlas. Group B is the sci-Plex drug screen. STACK reads single-cell RNA counts, so the bulk dataset from the same meeting was left out. We did not retrain the model. We downloaded the published weights and asked the model to rewrite one group of cells while looking at another group.

## What we downloaded

The organoid data are the raw counts from the Uzquiano, Kedaigle, and Arlotta cortical organoid atlas (study SCP1756 on the Broad Single Cell Portal). The public study bucket provided the count matrices. We used those raw counts, and we left the already-normalized tables alone, because STACK applies its own scaling. Cell-type labels came from the cluster files that sit alongside the counts.

The sci-Plex data are sample GSM4150378, the screen of A549, MCF7, and K562 cells treated with a library of compounds. The cell table and the gene names came from GEO. The count file on GEO would not download from this machine, so the counts came from the scPerturb release of the same experiment on Figshare. That release had lost the gene names, so we put the GEO names back on, and we checked that a few well-known genes landed on the right columns.

The model is the published Stack-Large-Aligned checkpoint from Hugging Face, together with the list of about 15,000 genes it was built to read.

## How the cells were divided

STACK looks at a fixed window of 512 cells. In the usual setup, about 333 of those cells are the example it is allowed to see, and about 179 are the cells it has to rewrite. We kept one window per group, taken as a fixed random sample of real cells, so the runs would finish on a CPU and would still be repeatable. The full atlas and the full screen are much larger than this.

For the organoids we kept the earliest and the latest ages that already had both a count matrix and a cell-type file, which were 23 days and 5 months. Within those ages we dropped labels that were not a real class. The example group is the class with the most cells at 23 days, which is apical radial glia, the early progenitor. The cells to rewrite are a different class that is present at both ages and is most abundant at 5 months, which is the intermediate progenitor. Each age contributed 333 radial glia. The 179 intermediate progenitors all come from 5 months, and the same 179 cells were rewritten twice, once with the young radial glia as the example and once with the older radial glia.

For sci-Plex the cells to rewrite are untreated MCF7 cells. The example cells are A549 cells, which is the setup from the meeting: show the model a treated lung line, and ask what the breast line would look like. MCF7 has untreated cells only at 24 hours, so 24 hours is the only exposure where both lines have a matched untreated group. Among the doses used at 24 hours, 10 nM is the dose where the largest number of compounds have at least 200 cells in both A549 and MCF7. At that dose we kept the three compounds with the most cells in the scarcer of the two lines: Ki8751, tranylcypromine, and GSK J1. They were chosen because both lines were well sampled, not because they were expected to be the strongest drugs. The model then ran four times on the same 179 untreated MCF7 cells, with the A549 example set to untreated cells, or to one of those three compounds.

Each run writes one file of new counts for those 179 cells. The organoid files are split by age. The sci-Plex files are split by compound.

## Cortical organoids

The rewritten intermediate progenitors still look like cortical progenitors. The average pattern of which genes are high and which are low agrees with the real intermediate progenitors, with a correlation of 0.87 when the example is 23-day radial glia and 0.86 when the example is 5-month radial glia. The two rewrites agree with each other at 0.94. The leading genes are the same ones that lead the real data, including MALAT1, and 65 of the 100 strongest genes are shared. The total counts per cell stay close to the measurement.

Cell by cell the match is looser, around 0.4 to 0.5. That is expected, because the model fills in a plausible count for each gene rather than copying the original cell. The plot of the cells makes the remaining difference obvious. Real intermediate progenitors sit between the two ages of radial glia. Cells rewritten from the 23-day example fall in with the young radial glia, and cells rewritten from the 5-month example fall in with the older radial glia. A few genes tell the same story. EOMES, a marker of intermediate progenitors, is present in the real cells and missing in the radial glia. The young example keeps it, and the older example mostly loses it. VIM, a progenitor gene, is turned up well above the real intermediate progenitors in both rewrites. TUBB3 is essentially absent in these measured cells and is added by the 23-day example.

![Average generated expression against the measured intermediate progenitors](figures/organoid_profile.png)

![Organoid cells and the rewritten profiles](figures/organoid_pca.png)

## sci-Plex

The sci-Plex measurement is single-nucleus RNA from a drug screen, and it looks very different from the organoid data. In these untreated MCF7 cells, familiar genes such as MALAT1, GAPDH, and ACTB are essentially missing, and the strongest genes are others, led by SACM1L. The rewrites do not recover that pattern. Agreement with the real untreated MCF7 cells is about 0.02 for every example we tried, and the 100 strongest genes in the rewrite and in the measurement do not overlap. The four rewrites do agree with each other, at a correlation of 0.99, so changing the A549 example from untreated cells to one of the three compounds barely moves the result. The small shift that does appear does not match the drug effect measured in MCF7, and it does not match the drug effect measured in A549. In this sample the two real cell lines also do not share a drug shift with each other, so there was little common drug signature for the example to carry across.

![Generated expression against the measured untreated MCF7 cells](figures/sciplex_profile.png)

![The generated drug shift compared with the measured shift in each cell line](figures/sciplex_shift.png)

## What the two runs say

On the organoid window, the published model stays inside a cortical progenitor program and lets the age of the radial glia pull the rewritten cells toward the example it was shown. On the sci-Plex window, showing it treated or untreated A549 cells does not produce a recognizable MCF7 profile, and it does not reproduce the measured drug effect in either line. Both readings come from one window of real cells, not from the whole atlas or the whole screen.
