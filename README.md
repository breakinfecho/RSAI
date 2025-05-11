# RSAI
## Introduction
    This study aims to enhance the dual-task synergy of large language models (LLMs) in customer-merchant review 
interpretation generation and survival prediction, thereby advancing intelligent decision-making technologies in 
vertical domains. As illustrated in Figure 2, our framework innovatively proposes a two-phase knowledge infusion 
strategy: the Supervised Learning phase (Section 3.1) facilitates domain knowledge transfer, while the Reinforcement 
Learning phase (Section 3.2) achieves capability-preference alignment. Through this phased optimization approach,
the framework systematically transforms general-purpose LLMs into specialized interpretive agents.

## Raw dataset format
{
    'input_text': { instruction } , 
    'output_text': { explanation} ,
    'input_filed_data':{
            review
            explanation
            rating
            restaurant_id
            survival
            features
       }
}

## SFT stage


## RL stage

