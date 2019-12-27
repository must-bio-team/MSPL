# MSPL: Multimodal Self-Paced Learning for multi-omics feature selection and data integration

This repository provides some codes for this paper. 

The code Version 1.0.

If you find it interesting and confused about the content, please contact me.

communication E-mail: yangziyi091100@163.com

## I . Abstract

Rapid advances in high-throughput sequencing technology have led to the generation of a large number of
multi-omics biological datasets. Integrating data from different omics provides an unprecedented opportunity to gain insight into disease mechanism from different perspectives. However, integrative analysis and predictive modeling from multi-omics data are facing three major challenges: i) heavy noises; ii) the high dimensions compared to the small samples; iii) data heterogeneity. Current multi-omics data integration approaches have some limitations and are susceptible to heavy noise. In this paper, we present MSPL, a robust supervised multi-omics data integration method that simultaneously identifies significant multi-omics signatures during the integration process and predicts the cancer subtypes. The proposed method not only inherits the generalization performance of self-paced learning, but also
leverages the properties of multi-omics data containing correlated information to interactively recommend high-confidence samples for model training. We demonstrate the capabilities of MSPL using simulated data and five multi-omics biological datasets, integrating up three omics to identify potential biological signatures, and evaluating the performance compared to state-of-the-art methods in binary and multi-class classification problems. Our proposed model makes multi-omics data integration more systematic and expands its range of applications.

If you find this code useful in your research then please cite:
```bash
@article{yang2019mspl,
  title={MSPL: Multimodal Self-Paced Learning for multi-omics feature selection and data integration},
  author={Yang, Zi-Yi and Xia, Liang-Yong and Zhang, Hui and Liang, Yong},
  journal={IEEE Access},
  year={2019},
  publisher={IEEE}
}
```

## II. Introduce about code

### i . The repository can be divided into three parts: 

1. Codes for simulated experiments.
2. Codes for benchmark multi-omics cancer datasets. (binary classification problem)
3. Codes for breast multi-omics cancer datasets. (multiple classification problem)

### ii . The compared methods:

Current supervised multimodal data integration approaches for predicting cancer subtypes and identifying significant multi-omics signatures can be classified as concatenation-based, ensemble-based, and knowledge-driven approaches.

We applied the logistic regression model/multinomial model with Elastic Net (EN) regularization [1], Random Forest (RF) [2] and Self-paced Learning (SPL) with L1 penalty [3] in the concatenation and ensemble frameworks. The compared methods include: concatenation-based methods (Concate\_EN, Concate\_ RF, and Concate\_SPL), ensemble-based methods (Ensemble\_EN, Ensemble\_RF, and Ensemble\_SPL) and DIABLO [4]. In this paper, we proposed a novel model for multi-omics data integration, termed as MSPL. 

### iii. The MSPL model:

The objective function of MSPL can be expressed as:
$$
\mathop{\min}_{\substack{\beta^{\left(j\right)},v^{\left(j\right)}\in\left[0,1\right]^n,\\j=1,2,\ldots,m}}E(\beta^{\left(j\right)},v^{\left(j\right)};\lambda^{(j)},\gamma^{\left(j\right)},\delta)=\sum_{j=1}^{m}\sum_{i=1}^{n}{v_i^{(j)}L\left(y_i,f^{\left(j\right)}\left(x_i^{\left(j\right)},\beta^{\left(j\right)}\right)\right)}+\sum_{j=1}^{m}{\lambda^{(j)}||\beta^{\left(j\right)}||_{1}} \\
-\sum_{j=1}^{m}\sum_{i=1}^{n}{\gamma^{\left(j\right)}v_i^{(j)}}-\delta\sum_{\substack{1\le k,j\le m,\\k\neq j}}{\left(v^{\left(k\right)}\right)^Tv^{\left(j\right)}},
$$

## III. Codes for simulated experiments

The codes for simulated experiments contain three parts:

1. Generating simulated data.
2. Train different data sets using training data sets and get the best solution for each model.
3. Calculate the prediction and feature selection performance. (We applied five indicators to evaluate the prediction performance: accuracy, sensitivity, specificity, recall, and AUC. To evaluate the feature selection performance, $\beta$-sensitivity and $\beta$-specificity are applied.)

Running "*\_simu.R" to perform the simulated experiments. Before running the code, it is need to set the path "setwd("D:/1-Simulation")". After that, running the code. For example:

```
Rscript 8-MSPL-simu.R
```

The defined of $\beta$-sensitivity and $\beta$-specificity are as follows:
$$
True Positive (TP)= \left|\beta.\ast\hat{\beta}\right|_0,   True Negative (TN)= \left|\bar{\beta}.\ast\bar{\hat{\beta}}\right|_0 \\
False Positive (FP)= \left|\bar{\beta}.\ast\hat{\beta}\right|_0 ,   False Negative (FN)=\ \left|\beta.\ast\bar{\hat{\beta}}\right|_0 \\
\beta-sensitivity= \frac{TP}{TP+FN}\\
\beta-specificity=\ \frac{TN}{TN+FP}
$$
where the $|\cdot|_0$ represents the number of non-zero elements in a vector. The logical not operators of  $\beta$ and $\hat{\beta}$ are $\bar{\beta}$ and $\bar{\hat{\beta}}$, respectively. And $.\ast$ is the element-wise product.

***Special comments:***

Since biological samples are complex and cannot be visualized, the choice of model parameters is a challenge. In order to make the model better select the samples in each class in the process of sample selection, we added the parameters of the selected sample size in the process of self-step learning.

## IV. Codes for benchmark multi-omics cancer data sets

Four benchmark multi-omics cancer datasets (mRNA, miRNA and DNA methylation) were obtained from [5]: Glioblastoma multi-forme (GBM), Kidney renal clear cell carcinoma (KRCCC), Lung squamous cell carcinoma (LSCC), Colon adenocarcinoma (COAD). Survival times were provided for each disease cohort by [5]. By using the median survival time, we dichotomized the samples into two classes in low and high survival times. A brief description of these four benchmark datasets is summarized in Table I.

Table I. The measurements of sample sizes and the number of features in each omics for four benchmark cancer datasets.

| Benchmark datasets | Samples (high/low) | mRNA  | miRNA | methylation |
| :----------------: | :----------------: | :---: | :---: | :---------: |
|       KRCCC        |    122 (61/61)     | 17665 |  329  |    24960    |
|        LSCC        |    106 (53/53)     | 12042 |  352  |    23074    |
|        GBM         |   213 (105/108)    | 12042 |  534  |    1305     |
|        COAD        |     92 (33/59)     | 17814 |  312  |    23088    |



The codes for simulated experiments contain three parts:

1. Pre-processing benchmark cancer datasets.
2. Train different data sets using training data sets and get the best solution for each model.
3. Calculate the prediction performance. (We applied five indicators to evaluate the prediction performance: accuracy, sensitivity, specificity, recall, and AUC.)

Running "*-bench.R" to perform the benchmark experiments. Before running the code, it is need to set the path "setwd("D:/2-Benchmark")". Moreover, some parameters need to adjust according to the specific problems. After that, running the code. For example:

```
Rscript 9-MSPL-bench.R
```

Intermediate result presentation, For example:

```
Starting the 1-th iteration.
The 1-th modality select 8samples.
The 2-th modality select 8samples.
The 3-th modality select 8samples.
Starting the 2-th iteration.
The 1-th modality select 8samples.
The 2-th modality select 12samples.
The 3-th modality select 11samples.
Starting the 3-th iteration.
The 1-th modality select 16samples.
The 2-th modality select 16samples.
The 3-th modality select 16samples.
...
```

***Special comments:***

For the benchmark cancer data experiment, we evaluate the classifier for the binary classification problem.

Since biological samples are complex and cannot be visualized, the choice of model parameters is a challenge. In order to make the model better select the samples in each class in the process of sample selection, we added the parameters of the selected sample size in the process of self-step learning.

## V. Codes for breast multi-omics cancer data set

We curated breast cancer multi-omics dataset (mRNA, miRNA and methylation) from the Cancer Genome Atlas (TCGA, data version 2015 11 01 for BRCA) in order to achieve a systems characterization of breast cancer subtypes with multiple omics. 

The raw data of breast cancer multi-omics can be download from the website: http://gdac.broadinstitute.org/runs/stddata__2015_11_01/data/BRCA/20151101/.

This dataset contains four subtypes of breast cancer: Luminal A (LumA), Luminal B (LumB), Her2-enriched (Her2) and Basal-like (Basal), which have been reported the most replicated subtypes of human breast cancer. The miRNA dataset was derived from two different Illumina technologies: The Illumina Genome Analyzer and the Illumina Hiseq. The methylation data was derived from two different platforms: the Illumina Methylation 27 and the Illumina 450K.

The codes for simulated experiments contain three parts:

1. Pre-processing breast cancer multi-omics dataset.
2. Train different data sets using training data sets and get the best solution for each model.
3. Calculate the prediction performance. (We applied five indicators to evaluate the prediction performance: accuracy, sensitivity, specificity, recall, and AUC.)

Running "*\_brac.R" to perform the benchmark experiments. Before running the code, it is need to set the path "setwd("D:/3-Breast")". Moreover, some parameters need to adjust according to the specific problems. After that, running the code. For example:

```
Rscript 9-MSPL_brac.R
```

Partial result presentation：

```
> perf.MVSPL
$Coef.mRNA
       [,1]                [,2]                   
  [1,] "mrna_ABCC11"       "-0.0243814271218362"  
  [2,] "mrna_AGR3"         "-0.0575991956876828"  
  [3,] "mrna_AKR1A1"       "-0.0237839509801553"  
  [4,] "mrna_AKR7A3"       "-0.0145826099658471"  
  [5,] "mrna_ALOX15B"      "-0.00171128246201879" 
  [6,] "mrna_AMY1A"        "0.0314640884955191"   
  [7,] "mrna_ANKRD37"      "-0.00104674226746144" 
  [8,] "mrna_ANKRD45"      "0.0108691104149298"   
  [9,] "mrna_ANO4"         "0.0108481887111139"   
 [10,] "mrna_ANXA8L2"      "0.0253604408490925"   
 ...
 $Coef.miRNA
       [,1]                   [,2]                   
  [1,] "mirna_hsa-let-7b"     "0.00648610316603627"  
  [2,] "mirna_hsa-let-7c"     "0.104315810731142"    
  [3,] "mirna_hsa-mir-1-2"    "-0.00533471987114328" 
  [4,] "mirna_hsa-mir-100"    "0.0193980789238908"   
  [5,] "mirna_hsa-mir-101-1"  "-0.114187671219588"   
  [6,] "mirna_hsa-mir-106b"   "0.0652289352080465"   
  [7,] "mirna_hsa-mir-10a"    "-0.00610670242225545" 
  [8,] "mirna_hsa-mir-10b"    "-0.0301647857895506"  
  [9,] "mirna_hsa-mir-1245"   "-0.000604381277121258"
 [10,] "mirna_hsa-mir-125a"   "0.015613554821614"   
 ...
 $Coef.meth
       [,1]                      [,2]                   
  [1,] "meth_A2ML1"              "-1.23127046096447"    
  [2,] "meth_ABCC3"              "-0.0342578433926931"  
  [3,] "meth_ABCG8;ABCG5"        "0.862207486855033"    
  [4,] "meth_ABTB1;PODXL2"       "-0.0882887040662381"  
  [5,] "meth_ACAP3"              "0.00532680412418839"  
  [6,] "meth_ACCN4"              "-0.00604362637967722" 
  [7,] "meth_ADAP1"              "-0.155742108124966"   
  [8,] "meth_ADIPOQ"             "1.75079005690609"     
  [9,] "meth_ADRBK2"             "0.0103044585323673"   
 [10,] "meth_AGA"                "-0.000200983462896094"
 ...
 $Perf.Train
Confusion Matrix and Statistics

          Reference
Prediction   1   2   3   4
         1 102   0   0   0
         2   0  40   0   0
         3   0   0 344   2
         4   0   0   0 122

Overall Statistics
                                          
               Accuracy : 0.9967          
                 95% CI : (0.9882, 0.9996)
    No Information Rate : 0.5639          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9946          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 1 Class: 2 Class: 3 Class: 4
Precision              1.0000  1.00000   0.9942   1.0000
Recall                 1.0000  1.00000   1.0000   0.9839
F1                     1.0000  1.00000   0.9971   0.9919
Prevalence             0.1672  0.06557   0.5639   0.2033
Detection Rate         0.1672  0.06557   0.5639   0.2000
Detection Prevalence   0.1672  0.06557   0.5672   0.2000
Balanced Accuracy      1.0000  1.00000   0.9962   0.9919

$Perf.Test
Confusion Matrix and Statistics

          Reference
Prediction   1   2   3   4
         1  75   1   0   0
         2   1  27   5   5
         3   0   0 169  19
         4   0   2  12  63

Overall Statistics
                                          
               Accuracy : 0.8813          
                 95% CI : (0.8444, 0.9121)
    No Information Rate : 0.4908          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.8206          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 1 Class: 2 Class: 3 Class: 4
Precision              0.9868  0.71053   0.8989   0.8182
Recall                 0.9868  0.90000   0.9086   0.7241
F1                     0.9868  0.79412   0.9037   0.7683
Prevalence             0.2005  0.07916   0.4908   0.2296
Detection Rate         0.1979  0.07124   0.4459   0.1662
Detection Prevalence   0.2005  0.10026   0.4960   0.2032
Balanced Accuracy      0.9918  0.93424   0.9051   0.8381
```

***Special comments:***

For the breast cancer multi-omics data experiment, we evaluate the classifier for the multiple classification problem.

Since biological samples are complex and cannot be visualized, the choice of model parameters is a challenge. In order to make the model better select the samples in each class in the process of sample selection, we added the parameters of the selected sample size in the process of self-step learning.



## VI. Reference:

[1] H. Zou and T. Hastie, “Regularization and variable selection via the elastic net,” Journal of the royal statistical society:  series B (statistical methodology), vol. 67, no. 2, pp. 301–320, 2005.

[2] A. Liaw, M. Wiener et al., “Classification and regression by randomforest,” R news, vol. 2, no. 3, pp. 18–22, 2002.

[3] M. P. Kumar, B. Packer, and D. Koller, “Self-paced learning for latent variable models,” in Advances in Neural Information Processing Systems, 2010, pp. 1189–1197.

[4] A. Singh, C. P. Shannon, B. Gautier, F. Rohart, M. Vacher, S. J. Tebbutt, and K.-A. Lˆe Cao, “Diablo: an integrative approach for identifying key molecular drivers from multi-omics assays,” Bioinformatics, 2019.

[5] B. Wang, A. M. Mezlini, F. Demir, M. Fiume, Z. Tu, M. Brudno, B. Haibe-Kains, and A. Goldenberg, “Similarity network fusion for aggregating data types on a genomic scale,” Nature methods, vol. 11, no. 3, p. 333, 2014.









