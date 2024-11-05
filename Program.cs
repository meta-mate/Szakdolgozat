using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
//using Microsoft.ML.OnnxRuntime.Gpu;
using Python.Runtime;
using System;


public class BertNodeValue : NodeValue {

    private string value;
    private BertUncasedBaseTokenizer tokenizer;
    private RunOptions runOptions;
    private InferenceSession session;

    static char[] punctuations = {'.', '!', '?', ';', ':', '-'};

    public BertNodeValue(string value, BertUncasedBaseTokenizer tokenizer, RunOptions runOptions, InferenceSession session) {
        this.value = value;
        this.tokenizer = tokenizer;
        this.runOptions = runOptions;
        this.session = session;
    }

    public string getValue(){
        return value;
    }

    public override string ToString(){
        return value;
    }

    public T deriveImplication<T>(List<T> nodeValues) where T : NodeValue{
        
        string inputSentence = "";
        for(int i = 0; i < nodeValues.Count; i++){
            BertNodeValue bertNodeValue = nodeValues[i] as BertNodeValue;
            if(bertNodeValue != null){
                inputSentence += bertNodeValue.getValue() + " ";
            }
        }

        inputSentence += value + " [MASK]";
        
        bool isPunctuation = false;
        foreach(char c in punctuations){
            if(value == c.ToString()) isPunctuation = true; 
        }
        if(!isPunctuation) inputSentence += ".";

        //Console.WriteLine(inputSentence);

        //Inferencing

        var tokens = tokenizer.Tokenize(inputSentence);
        var encoded = tokenizer.Encode(tokens.Count(), inputSentence);

        var bertInput = new BertInput()
        {
            InputIds = encoded.Select(t => t.InputIds).ToArray(),
            AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
            TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
        };

        using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                new long[] { 1, bertInput.InputIds.Length });

        using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                new long[] { 1, bertInput.AttentionMask.Length });

        using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                new long[] { 1, bertInput.TypeIds.Length });

        var inputs = new Dictionary<string, OrtValue>
        {
            { "input_ids", inputIdsOrtValue },
            { "input_mask", attMaskOrtValue },
            { "segment_ids", typeIdsOrtValue }
        };

         
        using var output = session.Run(runOptions, inputs, session.OutputNames);


        int GetMaxValueIndex(ReadOnlySpan<float> span)
        {
            float maxVal = span[0];
            int maxIndex = 0;
            for (int i = 1; i < span.Length; ++i)
            {
                var v = span[i];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        var startLogits = output[0].GetTensorDataAsSpan<float>();
        int startIndex = GetMaxValueIndex(startLogits);

        var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
        int endIndex = GetMaxValueIndex(endLogits);

        var predictedTokens = tokens
                        .Skip(startIndex)
                        .Take(endIndex + 1 - startIndex)
                        .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                        .ToList();

        // Print the result.
        Console.WriteLine(String.Join(" ", predictedTokens));

        string newValue = "";
        if(predictedTokens.Count > 0) newValue = predictedTokens[0];
        else newValue = "nooutput";
        NodeValue result = new BertNodeValue(newValue, tokenizer, runOptions, session);

        return (T)result;
    }
}

public struct BertInput
{
    public long[] InputIds { get; set; }
    public long[] AttentionMask { get; set; }
    public long[] TypeIds { get; set; }
}


namespace MyApp
{
    internal class Program
    {
        static void Main(string[] args)
        {

            var sentence = "Bob Dylan is from Duluth, Minnesota and is an American [MASK].";

            string[] inputSentence = {"this", "was", "a", "great", "movie"};

            var tokenizer = new BertUncasedBaseTokenizer();
            var tokens = tokenizer.Tokenize(sentence);
            var encoded = tokenizer.Encode(tokens.Count(), sentence);

            var bertInput = new BertInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };

            var modelPath = @".\bert-base-uncased.onnx";

            using var runOptions = new RunOptions();
            using var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            using var session = new InferenceSession(modelPath, gpuSessionOptions);
            //using var session = new InferenceSession(modelPath);

            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                new long[] { 1, bertInput.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                    new long[] { 1, bertInput.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                    new long[] { 1, bertInput.TypeIds.Length });

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "input_mask", attMaskOrtValue },
                { "segment_ids", typeIdsOrtValue }
            };

            
            using var output = session.Run(runOptions, inputs, session.OutputNames);


            int GetMaxValueIndex(ReadOnlySpan<float> span)
            {
                float maxVal = span[0];
                int maxIndex = 0;
                for (int i = 1; i < span.Length; ++i)
                {
                    var v = span[i];
                    if (v > maxVal)
                    {
                        maxVal = v;
                        maxIndex = i;
                    }
                }
                return maxIndex;
            }

            var startLogits = output[0].GetTensorDataAsSpan<float>();
            int startIndex = GetMaxValueIndex(startLogits);

            var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
            int endIndex = GetMaxValueIndex(endLogits);

            var predictedTokens = tokens
                            .Skip(startIndex)
                            .Take(endIndex + 1 - startIndex)
                            .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                            .ToList();

            // Print the result.
            Console.WriteLine(String.Join(" ", predictedTokens));





            /*
            PatternReader<BertNodeValue> patternReader = new PatternReader<BertNodeValue>();

            for(int i = 0; i < inputSentence.Length; i++){
                patternReader.interpretation(new BertNodeValue(inputSentence[i], tokenizer, runOptions, session));
            }

            Console.WriteLine(patternReader.toStringNodeList(false));
            */

            /*
            int[] pattern = {1, 0, 1};

            int predictedAmount = 0;
            for (int i = 0; i < pattern.Length; i++){
                int lastChange = patternReader.interpretation(2, pattern[i]);
                if (lastChange == 0 && i > 0) predictedAmount++;
            }

            Console.WriteLine(
                "predicted amount " + predictedAmount + "/" + pattern.Length + "\n" +
                patternReader.toStringNodeList(false)
                );

            */
        }
    }
}