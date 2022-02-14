import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class WekaKdTreeDataset {

    public static void main(String[] args) throws Exception {

        String fileName = "testData3";

        InstanceManager manager = new InstanceManager(fileName);
        Instances train = manager.getTrain();
        Instances test = manager.getTest();
        Instances all = manager.getAll();
        manager.printInstances();

        MyAlgorithm classifier = new MyAlgorithm(3);
        String[] options = new String[1];
        options[0] = "-K";
        classifier.setOptions(options);

        classifier.buildClassifier(all);

        //plnenie instancii
        int size = 3;
        Instances baseInstances = new Instances("Test", manager.getALlAttributes(), 1);
        baseInstances.setClassIndex(baseInstances.numAttributes() - 1);

        double[] instanceValue = new double[size];
        instanceValue[0] = 7;
        instanceValue[1] = 2;
        instanceValue[2] = 1;
        Instance instance = new DenseInstance(1d, instanceValue);
        baseInstances.add(instance);
        double v = classifier.classifyInstance(instance);
        System.out.println(v);
        System.out.println("----------------------");

        double[] doubles = classifier.distributionForInstance(baseInstances.firstInstance());
        for (int i = 0; i < doubles.length; i++) {
            System.out.println(doubles[i]);
        }

        EvaluationManager evaluationManager = new EvaluationManager(classifier, test);
    }
}
