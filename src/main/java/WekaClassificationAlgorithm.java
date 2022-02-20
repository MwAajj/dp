import dataset.ArffManager;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class WekaClassificationAlgorithm {

    public static void main(String[] args) throws Exception {
        //kriterium - shannova entropia, klassification error, giny index, fuzzy rozhodovacia strom,
        IBk classifier;
        int k = 3;

        //---------------------------KNN---------------------------
        classifier = new IBk(k);
        String[] options = {"-K", "5", "-I"};

        classifier.setOptions(options);

        // ---------------------------NaiveBayes---------------------------
        //classifier = new NaiveBayes();

        // ---------------------------Decision trees J48---------------------------
        //classifier = new J48();

        //classifier = new MyAlgorithm();
        //DatasetManager dataset = new DatasetManager(true);
        String fileName = "testData4Fuzzy";

        InstanceManager manager = new InstanceManager(fileName);

        Instances all = manager.getAll();
        Instances test = manager.getTest();
        Instances train = manager.getTrain();

        //manager.getNewInstance();
        train.setClassIndex(2);
        classifier.buildClassifier(train);
        Instances baseInstances = new Instances("Test", manager.getALlAttributes(), 1);
        baseInstances.setClassIndex(baseInstances.numAttributes() - 1);

        double[] instanceValue = new double[3];
        instanceValue[0] = 7;
        instanceValue[1] = 2;
        instanceValue[2] = 1;
        Instance instance = new DenseInstance(1d, instanceValue);
        baseInstances.add(instance);
        double v = classifier.classifyInstance(baseInstances.firstInstance());
        System.out.println(v);
        System.out.println("-------------------------");

        double[] doubles = classifier.distributionForInstance(baseInstances.firstInstance());
        for (int i = 0; i < doubles.length; i++) {
            System.out.println(doubles[i]);
        }


        EvaluationManager evaluation = new EvaluationManager(classifier, test, train);

        //epoch for bigger data
        //batch - zhluk dat

        //filtracia dat
        //normalizacia vstupnych dat preco sa to robi
        //aby mal lepsiu formu ucenia algoritmus
    }
}
