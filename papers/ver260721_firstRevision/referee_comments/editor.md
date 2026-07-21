# Editor and Referee's comment

## Editor's Comment
 
>Dear Dr Moon,
>
>Your manuscript, "Radionuclide Identification for Plastic ScintillationDetectors by Backward Cumulative Channel Sum", has now been assessed. 
>
>We invite you to revise your paper, carefully addressing the comments from the reviewers and the editor. Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained.  When your revision is ready, please submit the updated manuscript and a point-by-point response. This will help us move to a swift decision.
> 
>Please note that if your manuscript uses any custom or bespoke computational tool or code, or reports a new algorithm, tool, software, or a pipeline (even if individual components are not new), the underlying code must be deposited in a recognised DOI-assigning repository (e.g. zenodo) and linked either from Methods or a dedicated Code Availability section. 
> 
>Editor Comments 
> 
>"The reviewers have highlighted major issues with the manuscript, specifically lack of quantitative validation, comparison with established methods and missing details regarding calibration, and quantitative evaluation. " 
>
>-Guy Garty 
> 
>We recommend submitting all revisions within the mentioned deadline. 
> 
>If you need more time, please contact us and include your submission ID. 
> 
>Kind regards, 
> 
>Pranjal Waghmare 
>Assistant Editor 
>Scientific Reports 
>Support contact: srep@nature.com 
>Submission ID: 2cf10123-75ae-402b-ba8e-530bd1dd4208 
  
</br>

## Reviewer 1

The manuscript presents a simple approach for radionuclide identification using plastic scintillation detectors, which could be of interest for radiation portal monitoring. However, there are several important aspects that are not clearly described and make it difficult to fully understand and reproduce the results. In particular, more information is needed about the reference spectra, the detector calibration, and how the fitting is evaluated. It would also be helpful if the authors discussed more clearly the limitations of the method and the conditions in which it can be applied. 

Attachment(s): 
• Download Reviewer 1 attachment 1 

### Contents in Attachments

The reviewed manuscript presents a methodology to identify three different isotopes from spectra obtained with plastic scintillators in radiation portals. The methodology is  based on the Backward Cumulative Channel Sum (BCCS) which transforms the measured spectrum to avoid fluctuations of the higher channel area. Once  the BCCS is performed, the authors apply two different calculations, either subtraction or normalization of the background, after which they use the non-negative least-squares method to reproduce the experimental spectrum from the reference spectra. 

The simplicity of this method makes it interesting for its use in radiation portal monitoring. However, there are different aspects that the authors should clarify regarding the methodology and experimental limitations. These aspects are directly related to the robustness of the method, the reproducibility of the work and its practical application in real situations. 

#### Major Comments 

1. Reference spectra: 
    The method described relies in the decomposition of real spectra. To do this, reference spectra are used. The manuscript lacks a description that explains how the reference spectra were obtained. No acquisition protocols, measurement conditions, processing of the obtained spectra, shape of the radioactive source, background effects etc.. were detailed. This is critical because the method accuracy relies on the representativeness of the reference spectra and makes the reproducibility of the study doubtful.  
</br>

2. Radioactive sources: 
   Following the same idea mentioned before, the manuscript does not provide much information about the type of radioactive sources used to obtain the measured spectra. It is not specified whether the sources are point-like or extended. The geometry of the experiment is not described and no other aspects (such as scattering or absorption) are explained, making it difficult to assess the validity of the method for real situations measuring through a radiation portal.   
</br>

3. Low-channel region: 
   The method relies on the data of the low-channel region. However, there is no explanation regarding what “low-channel region” is. There is no threshold or a criterion to decide what is low or what is high. There is no physical explanation, or examples, to justify the truncation of the spectrum.  It is not clear whether this could be arbitrarily considered or dependent on acquisition parameters, such as the gain settings of the electronic hardware. For example, the gain can compress the major part of the spectrum into low channels or even misrepresent the high-energy contributions. It is not clear, for example, to what extent the photopeaks of Co-60 are included or excluded of the low-channel region. In my opinion, these details should be explained to characterize the system for reproducibility.  
</br>

4. Energy calibration and resolution 
   Considering the previous comment, the absence of energy calibration and detector characterization leave the manuscript with brief and vague descriptions. All analyses are performed in channels without establishing a relationship between channel number and physical energy. I understand that it makes the method simpler and even easier to apply without the calibration step, but lacks the description of the range of energies that the low-channel region includes, again, this should be noted for reproducibility reasons and comparison purposes. In addition, the manuscript mentions the “low resolution of plastic scintillators” but fails to give a numerical example for the reader comprehension of what the authors consider low.  
</br>

5. Experimental Validation 
   The manuscript presents a method useful for radiation portal monitoring but the experimental validation is very simple and not representative of a situation that usually includes absorption, shielding, scattering and complex geometries in variable backgrounds when radionuclides are mixed with other materials and loaded on vehicles. The method should be described to understand under which conditions it is considered adequate or useful. Or, in the other hand, show the performance and applicability of this method in a real situation.  
</br>

6. Evaluation of the NNLS fitting
   The manuscript mentions that the evaluation of the NNLS spectral decomposition is in agreement between the measured and the fitted spectra (obtained using a combinations of reference spectra) primarily through “visual agreement”. Even if the qualitative evaluation could be useful in some situations, the study does no provide any quantitative metric that assesses the quality of the fitting method. The absence of the goodness of the fit (via X2, C-stats, residual analysis, or other) and relying on the visual capacity is not sufficiently rigorous to demonstrate the accuracy of this method. Spectra with very low resolution and counts can appear similar to very different isotopes. In addition, even more important is the fact that, when using reconstruction methods, different combinations of reference spectra can obtain the measured spectra. The lack of a goodness of the fit makes comparison with other methods very difficult or impossible. Therefore, it is very difficult to evaluate that the proposed method improves radionuclide identification. No uncertainties of the radionuclide contribution of the coefficients of the fit (αx) are reported. 


Overall, this study proposes an interesting method, but the issues mentioned above are sufficiently significant and need to be addressed. Therefore I would advise a major revision before it can be considered for publication.  

</br>
## Reviewer 2 

No extra comments for the review process. 

</br>

## Reviewer 3 

This manuscript addresses a highly relevant and practically important challenge in radiation detection, namely the reliable identification of radionuclides using plastic scintillation detectors under low signal-to-background conditions. Given the widespread deployment of these detectors in radiation portal monitoring systems due to their cost-effectiveness, robustness, and high efficiency, improving their analytical performance without requiring hardware modifications is of significant interest to both the scientific community and applied security domains. The proposed BCCS-based spectral processing framework offers a promising and conceptually straightforward approach to enhancing spectral interpretability and radionuclide separability, which is critical for real-world monitoring scenarios. Overall, the work is well motivated and targets a problem of clear operational and scientific importance. 
 
The authors provide a clear and candid discussion of the method’s limitations, acknowledging dependencies on reference spectra quality, environmental stability, and background representativeness. Their identification of these factors, along with suggested future improvements, reflects a careful and balanced assessment that strengthens the overall credibility of the study. 
 
My main concern with the manuscript is that, although it claims that the proposed method enhances spectral separability and radionuclide identification, it does not provide sufficient quantitative results to support these claims. Furthermore, the manuscript does not include comparisons with alternative methods. In addition, the approaches used to evaluate the reliability of the results appear somewhat heuristic and are not grounded in established statistical frameworks, such as those described in ISO 11929. 
 
For these reasons, I recommend that the authors strengthen the manuscript by comparing the sensitivity of the proposed method with at least two commonly used approaches: 

1. Least-squares fitting using the original (untransformed) spectra, where the background spectrum is included as a template rather than subtracted prior to analysis.
2. An energy window method in which the windows correspond to the dominant regions of Am-241, Cs-137, and Co-60, with an additional high-energy window used to estimate the background level.
 
Minor comments: 
 
The structure of the manuscript is somewhat unconventional. The authors should consider presenting the Methods section before the Results section. 
 
The advantages of the two background compensation methods are described in both the subsections “Transformation of reference radionuclide spectra” and “Background compensation for mixed-source spectra.” These should be consolidated, preferably in the Methods section. 
 
Due to the fixed scaling of the y-axis, Figure 4 is difficult to read.
 
The y-axis of Figure 5 should be labeled. 
 
The measurement times for the analyzed spectra, background spectra, and template spectra are not clearly defined and should be explicitly stated.