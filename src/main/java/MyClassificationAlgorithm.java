import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MyClassificationAlgorithm {

    public static void main(String[] args) throws Exception {

        String fileName = "testData4Fuzzy";

        InstanceManager manager = new InstanceManager(fileName);
        Instances train = manager.getTrain();
        Instances test = manager.getTest();
        Instances all = manager.getAll();
        manager.printInstances();

        MyAlgorithm classifier = new MyAlgorithm(5);
        String[] options = new String[2];
        options[0] = "-K";
        options[1] = "-F";
        classifier.setOptions(options);

        classifier.buildClassifier(all);

        //plnenie instancii
        int size = 3;
        Instances baseInstances = new Instances("Test", manager.getALlAttributes(), 1);
        baseInstances.setClassIndex(baseInstances.numAttributes() - 1);

        double[] instanceValue = new double[size];
        instanceValue[0] = 3;
        instanceValue[1] = 2;
        instanceValue[2] = 1;
        Instance instance = new DenseInstance(1d, instanceValue);
        baseInstances.add(instance);
        double v = classifier.classifyInstance(instance);
        System.out.println("\n\n--------------------------------------------");
        System.out.println("Instance: " + instance + " belongs to class " + v);
        System.out.println("--------------------------------------------\n\n");

        double[] doubles = classifier.distributionForInstance(baseInstances.firstInstance());
        for (int i = 0; i < doubles.length; i++) {
            System.out.println(doubles[i]);
        }

        EvaluationManager evaluationManager = new EvaluationManager(classifier, test);
    }
}
