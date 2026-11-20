## What is SpokenWOZ?

SpokenWOZ is a large-scale multi-domain speech-text dataset for spoken task-oriented dialogue modeling, which consists of 203k turns, 5.7k dialogues and 249 hours audios from realistic human-to-human spoken conversations.

## Why SpokenWOZ?

The majority of existing TOD datasets are constructed via writing or paraphrasing from annotators rather than being collected from realistic spoken conversations. The written TDO datasets may not be representative of the way people naturally speak in real-world conversations, and make it difficult to train and evaluate models that are specifically designed for spoken TOD. Additionally, the robustness issue, such as ASR noise, also can not be fully explored using these written TOD datasets. Different exsiting spoken TOD datasets, we introduce common spoken characteristics in SpokenWOZ, such like word-by-word processing and commonsense in spoken language. SpokenWOZ also includes cross-turn detection and reasoning slot detection as new challenges to better handle these spoken characteristics.

## Getting Started

The data is split into training, development, and unreleased test sets. 

You can download a copy of the dataset (distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license): https://spokenwoz.github.io/SpokenWOZ-github.io/

Once you have built a model that works to your expectations on the dev set, you can submit it to [Wentao Ma](https://spokenwoz.github.io/SpokenWOZ-github.io/mawentao.mwt@alibaba-inc.com) to get official scores on a hidden test set. To preserve the integrity of test results, we do not release the test set to the public. Instead, we request you to submit your model so that we can run it on the test set for you.

## Data structure

There are 5,700 dialogues ranging form single-domain to multi-domain in SpokenWOZ. The test sets contain 1k examples.

Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues. Each dialogue consists of a goal, multiple user and system utterances, dialogue state, dialogue act, corresponding audio and ASR transcription. 

The file name of the audio is consistent with the id of the dialogue, for example, the corresponding audio file for MUL0032 is MUL0032.wav.

The dialogue goal for each dialogue is recorded in the "goal" field. The dialogue goal holds the fields involved in the dialogue as well as the slots involved and the corresponding values.

The dialogue state for each dialogue is recorded in the "metadata" field in every turn the same as MultiWOZ 2.1.  The  state have two sections: semi, book. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain. The joint accuracy metrics includes ALL slots.

The dialogue act for each dialogue is recorded in the "dialogue_act" and "span_info" field in every turn:

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "dialogue_act": {
        "$act_name": [
          [
            "$slot_name",
            "$action_value"
          ]
        ]
      },
      "span_info": [
        [
          "$act_name"
          "$slot_name",
          "$action_value"
          "$start_charater_index",
          "$exclusive_end_character_index"
        ]
  }
}
```

The ASR transcription for each dialogue is recorded in the "words" field in every turn.  

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "words": [
        {
        "$word_context": "$word",
        "$begin_time": "$begintime",
        "end_time": "$endtime",
        "channel_id": "$channel",
        "word_index": "$index",
        }
  }
}
```



# Citation

```
@article{si2023spokenwoz,
  title={SpokenWOZ: A Large-Scale Speech-Text Dataset for Spoken Task-Oriented Dialogue in Multiple Domains},
  author={Si, Shuzheng and Ma, Wentao and Wu, Yuchuan and Dai, Yinpei and Gao, Haoyu and Lin, Ting-En and Li, Hangyu and Yan, Rui and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2305.13040},
  year={2023}
}
```

