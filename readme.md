# ARC-AGI-2 results
## Training Loss
![](./AbstractHRM/saved/img/readme/Losses_tet.png)
## Evaluation Loss
![](./AbstractHRM/saved/img/readme/LossesEval.png)
## Training Accuracy
![](./AbstractHRM/saved/img/readme/Accuracy_tet.png)
## Evaluation Accuracy
![](./AbstractHRM/saved/img/readme/AccuracyEval.png)

<br>

# Latent values decoded:

<style>
    .row {
        display: flex;

        p {
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            text-align: center;
            width: 0;
        }
    }

    .latent-table {
        table {
            width: 100%;
            table-layout: fixed;
            margin-left: 2em;
        }

        th {
            border: none;
        }

    }
</style>

<div class="latent-table">

<table>
<tr>
<th>Input</th><th>Target</th><th>Output</th><th>Combined</th><th>Embed</th>
</tr>
</table>

<div class="row">
<p>Transformer</p>
<img src="./AbstractHRM/saved/img/readme/Transformer_latent.png" alt="">
</div>
<div class="row">
<p>TRM</p>
<img src="./AbstractHRM/saved/img/readme/TRM_latent.png" alt="">
</div>
<div class="row">
<p>AbstractHRM</p>
<img src="./AbstractHRM/saved/img/readme/AbstractHRM_latent.png" alt="">
</div>
</div>
<br>

As we go from Transformer to TRM to AbstractHRM, the less is the embedding and the example grid combiner close to the solution, which means the more is the thinking module utilized.

<br>

# Losses in output synchronization:

Checking if model <b>A</b> can learn model <b>B</b>'s behaviour better while model <b>B</b> is learning model <b>A</b>'s behaviour. If it does, then it means that model <b>A</b> has more adaptive freedom, while model <b>B</b> has more of an inductive bias, a fixed characterstic that cannot adapt.

### AbstractHRM vs Transformer
![](./AbstractHRM/saved/img/readme/AbstractHRM_Transformer.png)
### AbstractHRM vs TRM
![](./AbstractHRM/saved/img/readme/AbstractHRM_TRM.png)

In both cases my model's loss decreases then stabilizes, while the other models loss increases. This means that my model can learn these models behaviour, while they can't learn my model's behavior. This indicates that my model has less of an inductive bias, and more adaptive capacity.

### Transformer vs TRM
![](./AbstractHRM/saved/img/readme/Transformer_TRM.png)

![](./AbstractHRM/saved/img/readme/Transformer_TRM_regular.png)

It is inconsistent which model is the winner. This indicates that there is very little difference in their adaptive capacity.

### AbstractHRM vs AbstractHRM
![](./AbstractHRM/saved/img/readme/AbstractHRM_AbstractHRM.png)

This further proves that models that are very similar in adaptive capacity show this kind of tendency, where both losses decrease then stabilize, since in this case there is no difference in these models, just their random initialization.