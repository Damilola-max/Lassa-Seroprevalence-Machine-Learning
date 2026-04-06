
/************************************************************************************
Full Analysis of the LASSA Fver Surveillance Data conducted in kwara state, Nigeria
*************************************************************************************/

  -- SECTION ONE: Diagnostic Performance & Failure 
  -- PCR vs. serology agreement?
  SELECT TOP 250
     pcr_Results,
     IgM_OD_Results,
     [IgG_OD_Results2],
  COUNT(*)      AS Total_Cases
  FROM lab_results
  WHERE PCR_Results IS NOT NULL
  AND IgM_OD_Results IS NOT NULL
  AND IgG_OD_Results2 IS NOT NULL
  GROUP BY pcr_results,IgM_OD_results, [IgG_OD_Results2];

  -- PCR +ve but IgM/IgG -ve?
  SELECT
     COUNT(*)    AS pcr_pos_igm_neg
  FROM lab_results
  WHERE PCR_Results = '320Kb (Positive)'
  AND IgM_OD_Results IN ('Negative', 'Strongly Negative');

  SELECT
     COUNT(*)    AS pcr_pos_igg_neg
  FROM lab_results
  WHERE PCR_Results = '320Kb (Positive)'
  AND IgM_OD_Results = ('Negative');

                   -- ratio of IgM +ve but pcr -ve?
                   SELECT TOP 250
                       COUNT(*)  AS IgM_pos_pcr_neg
                   FROM lab_results
                   WHERE IgM_OD_Results IN ('Positive','Strongly Positive')
                   AND PCR_Results = 'No Kb (Negative)';

  -- what is the correlation between IgM vs. pcr?
  SELECT
  pcr_results,
      AVG(TRY_CAST(IgM_OD_Values AS DECIMAL(10,2))) AS avg_igm_od,
      MIN(TRY_CAST(IgM_OD_Values AS DECIMAL(10,2))) AS MIN_igm_od,
      MAX(TRY_CAST(IgM_OD_Values AS DECIMAL(10,2))) AS MAX_igm_od
  FROM lab_results
  WHERE TRY_CAST(IgM_OD_Values AS DECIMAL(10,2)) IS NOT NULL 
  GROUP BY PCR_Results;

  /*********************************************************************
  -- SECTION TWO: Temporal Diagnostic Window
   *********************************************************************/
   
        -- what are the days before symptoms onset? (no date table so track progression by on IgM & IgG values)
        SELECT TOP 250
        Patient_ID,
        CASE
            WHEN pcr_results = '320kb (Positive)'
            AND IgM_OD_results IN('Negative','Strongly Negative') THEN 'Early'
            WHEN pcr_results = '320kb (Positive)'
            AND IgM_OD_results IN('Positive','Strongly Positive') THEN 'Mid'
            WHEN IgM_OD_results IN('Positive', 'Strongly Positive')
            AND [IgG_OD_Results2] = 'Positive' THEN 'Late'
            WHEN [IgG_OD_Results2] = 'Positive'
            AND pcr_results = 'Negative' THEN 'Past'
        ELSE 'Unclassified'
        END AS infection_stage
        FROM lab_results
        WHERE IgM_OD_Values IS NOT NULL
        AND IgG_OD_Values IS NOT NULL
        ORDER BY infection_stage; 
        
        
                                       -- OR --

                                      /** use the antibody levels as time signal to get output **/
                                      -- Do higher IgG values indicate late stage of infection?

                                     SELECT TOP 250
                                     Patient_ID,
                                     IgM_OD_Values,
                                     IgG_OD_Values,
                                     CASE
                                        WHEN IgM_OD_Values > 0.17 AND IgG_OD_Values <= 0.1825 THEN 'Early Infection'
                                        WHEN IgM_OD_Values > 0.17 AND IgG_OD_Values > 0.1825 THEN 'Active Infection'
                                        WHEN IgM_OD_Values <= 0.17 AND IgG_OD_Values > 0.1825 THEN 'Late/Past Infection'
                                        ELSE 'Unclear'
                                        END AS infection_stage
                                     FROM lab_results
                                     WHERE IgM_OD_Values IS NOT NULL
                                     AND IgG_OD_Values IS NOT NULL
                                     ORDER BY infection_stage ASC;

        
        -- PCR positivity over time? --

       SELECT TOP 250
          Infection_stage,
          COUNT (*)   AS total_cases,
          SUM( pcr_positive_flag)   AS Pcr_positive,
          ROUND(100.0 * SUM( pcr_positive_flag) / COUNT(*),2) AS pcr_positivity_rate
       FROM(
       SELECT TOP 250
         CASE
            WHEN pcr_results = '320Kb (Positive)' AND TRY_CAST(IgM_OD_Values AS FLOAT) < 0.17 
            AND TRY_CAST(IgG_OD_Values AS FLOAT) < 0.1825 THEN 'Early Stage' 
            WHEN pcr_results = '320Kb (Positive)' AND TRY_CAST(IgM_OD_Values AS FLOAT) >= 0.17 
            AND TRY_CAST(IgG_OD_Values AS FLOAT) < 0.1825 THEN 'Mid Stage' 
            WHEN TRY_CAST(IgM_OD_results AS FLOAT) >= 0.15 AND TRY_CAST([IgG_OD_Results2] AS FLOAT) >= 0.15 THEN 'Late Stage'
            WHEN TRY_CAST([IgG_OD_Results2] AS FLOAT) >= 0.15 AND pcr_results = 'Negative' THEN 'Past Infection'
       ELSE 'Unclassified'
       END AS Infection_stage, 
            CASE
            WHEN pcr_results = '320Kb (Positive)' THEN 1 ELSE 0 END AS pcr_positive_flag
       FROM lab_results
       ) AS x
       GROUP BY Infection_stage;


        /*********************************************************************
         -- SECTION THREE: Multimodal Integration Gaps
        *********************************************************************/

        -- what are the missed cases by single method maybe using pcr alone or IgM?
        SELECT
           COUNT(*)   AS total_cases,
           SUM(CASE WHEN pcr_results = '320Kb (Positive)' THEN 1 ELSE 0 END) AS pcr_detected,
           SUM(CASE WHEN IgM_OD_results IN ('Positive', 'Strongly Positive') THEN 1 ELSE 0 END)  AS IgM_detected
        FROM(
        SELECT TOP 250*
        FROM lab_results
        )  AS sub;

        -- what are the cases missed if only pcr was used?
        SELECT TOP 250
           COUNT(*)   AS missed_by_pcr
        FROM lab_results
        WHERE PCR_Results = 'No Kb (Negative)'
        AND IgM_OD_Results IN ('Positive', 'Strongly Positive');

       
       /*********************************************************************
         -- SECTION FOUR: QUANTITATIVE OD ANALYSIS
        *********************************************************************/

        -- what is the OD threshold vs. PCR 
        SELECT
          CASE 
            WHEN IgM_OD_Values < 0.15 THEN 'Low'
            WHEN IgM_OD_Values < 0.5 THEN 'Medium'
            ELSE 'High'
        END AS IgM_level,
        COUNT(*) AS total,
        SUM(CASE WHEN pcr_results = '320Kb (Positive)' THEN 1 ELSE 0 END) AS pcr_positive
     FROM(
        SELECT TOP 250*
        FROM lab_results
        ) AS Sub
        GROUP BY 
           CASE
             WHEN IgM_OD_Values < 0.15 THEN 'Low'
             WHEN IgM_OD_Values < 0.5 THEN 'Medium'
             ELSE 'High'
             END
             ORDER BY IgM_level;

        -- Determine the IgG + IgM combined effect to PCR?
        SELECT
        pcr_results,
        AVG(TRY_CAST(IgM_OD_Values  AS DECIMAL(10,2))) AS avg_IgM,
        AVG(TRY_CAST(IgG_OD_Values  AS DECIMAL(10,2))) AS avg_IgG,
        AVG(
            TRY_CAST(IgM_OD_Values  AS DECIMAL(10,2)) + TRY_CAST(IgG_OD_Values  AS DECIMAL(10,2))
            ) AS combined_score
        FROM(
             SELECT TOP 250*
             FROM lab_results
             WHERE PCR_Results IS NOT NULL
             )AS sub
             GROUP BY PCR_Results;
            
        

        /*********************************************************************
         -- SECTION FIVE:Outcome Prediction Signals
        *********************************************************************/
        -- what is the correlation between lab results & hospitalization?
        SELECT TOP 250
        l.patient_ID,
        l.pcr_results,
        l.IgG_OD_Results2,
        l.IgM_OD_results,
        COUNT(*)   AS total,
        SUM(CASE WHEN t.Hospitalized = 1 THEN 1 ELSE 0 END)  AS Hospitalized_cases
        FROM lab_results l
        JOIN treatment_outcomes t ON l.Patient_ID = t.Patient_ID
        GROUP BY l.PCR_Results,l.Patient_ID,l.IgG_OD_Results2, l.IgM_OD_Results;

        -- what is the correlation between lab results & recovery?
        SELECT TOP 250
        l.pcr_results,
        l.IgM_OD_results,
        l.IgG_OD_results2,
        SUM(CASE WHEN t.fully_recovered = 1 THEN 1 ELSE 0 END) recovered,
        SUM(CASE WHEN t.fully_recovered = 0 THEN 1 ELSE 0 END) not_recovered
        FROM lab_results l 
        JOIN treatment_outcomes t ON l.Patient_ID = t.Patient_ID
        GROUP BY l.PCR_Results, l.IgM_OD_Results,l.IgG_OD_Results2;



        /*********************************************************************
         -- SECTION SIX:Exposure vs. lab confirmation?
        *********************************************************************/
         SELECT TOP 250
         p.patient_ID,
         p.rodent_contact_6m,
         p.Food_Open_Storage,
         l.pcr_results
         FROM patient_core p
        JOIN lab_results l
        ON p.patient_ID = l.patient_ID
        ORDER BY p.rodent_contact_6m, l.pcr_results, p.Food_Open_Storage ASC;                  
       
        /*********************************************************************
         -- SECTION SEVEN: Cross reactivity
        *********************************************************************/
       SELECT TOP 250
       p.state,
       COUNT(*)  AS silent_cases
       FROM lab_results l
       JOIN patient_core p ON l.Patient_ID = p.patient_ID
       WHERE l.IgG_OD_Results2 = 'Positive'
       AND l.PCR_Results = 'No Kb (Negative)'
       GROUP BY p.State;

       /*********************************************************************
         -- SECTION EIGHT: DIAGNOSTIC ALGORITHM CONSTRUCTION 
        *********************************************************************/
        SELECT TOP 250
        patient_ID,
          CASE
            WHEN pcr_results = '320Kb (Positive)' THEN 'Confirmed'
            WHEN igM_OD_results IN ('Positive', 'Strongly Positive') THEN 'Probable'
            WHEN IgG_OD_Results2 = 'Positive' THEN 'Past Infection'
            ELSE 'Negative'
        END AS case_classification
        FROM lab_results;

        /*********************************************************************
         -- SECTION NINE: FEATURE ENGINEERING 
        *********************************************************************/
        -- Combined Antibody score
        SELECT TOP 250
        patient_ID,
        (IgM_OD_results + IgG_OD_results2)   AS antibody_score
        FROM lab_results;

        --Diagnostic conflict flag
        SELECT TOP 250
        patient_ID,
        CASE
        WHEN pcr_results = 'positive'  AND IgM_OD_Results IN ('Negative', 'Strongly Negative') THEN 1 ELSE 0 
        END AS diagnostic_conflict
        FROM lab_results;




           