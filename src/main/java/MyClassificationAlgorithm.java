import classifier.MyAlgorithm;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MyClassificationAlgorithm {

    public static void main(String[] args) throws Exception {

        String fileName = "testObraz";

        InstanceManager manager = new InstanceManager(fileName);
        Instances train = manager.getTrain();
        Instances test = manager.getTest();
        Instances all = manager.getAll();
        manager.printInstances();

        MyAlgorithm classifier = new MyAlgorithm();
        String[] options = {"-K", "3", "-F", "2"};
        classifier.setOptions(options);

        classifier.buildClassifier(all);

        //plnenie instancii
        int size = 3;
        Instances baseInstances = new Instances("Test", manager.getALlAttributes(), 1);
        baseInstances.setClassIndex(baseInstances.numAttributes() - 1);

        double[] instanceValue = new double[size];
        instanceValue[0] = 5;
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
            System.out.println("Probability of class" + i +  " is " + doubles[i]);
        }
        System.out.println("\n");
        EvaluationManager evaluationManager = new EvaluationManager(classifier, test, train);
    }
}
