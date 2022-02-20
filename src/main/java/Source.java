import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import structure.kdtree.KdTree;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

public class Source
{
    public static void main(String[] args) throws Exception {
        DatasetManager datasetManager  = new DatasetManager("covidFull", "covidFull", 0);
        int k = 3;
        KDTree kdTree = new KDTree();

        //IBk classifier = new IBk(3);
        //String[] options = {"-K", "3", "-F", "2"};
        String[] options = {"-K", "10", "-H", "2"};
        MyAlgorithm classifier = new MyAlgorithm();
        classifier.setOptions(options);
        InstanceManager manager = new InstanceManager("covidFull", 0);


        Instances all = manager.getAll();
        Instances test = manager.getTest();
        Instances train = manager.getTrain();

        kdTree.setInstances(all);

        //System.out.println(all.firstInstance());
       // System.out.println(kdTree.nearestNeighbour(all.firstInstance()));
        train.setClassIndex(train.numAttributes() - 1);

        classifier.buildClassifier(train);
        /*for(Instance instance : all) {
            System.out.println("----------------------------------------------------------------------------------------------");
            System.out.println(instance);
            System.out.println("\n\n\n");
            classifier.help(instance);
            System.out.println("----------------------------------------------------------------------------------------------");
        }

        System.out.println("KASUBA");

        for(Instance instance : all) {
            System.out.println("----------------------------------------------------------------------------------------------");
            System.out.println(instance);
            System.out.println("\n\n\n");
            Instances instances = kdTree.kNearestNeighbours(instance, k);
            for (Instance instance1 : instances){
                System.out.println(instance1);
            }
            System.out.println("----------------------------------------------------------------------------------------------");

        }*/



        EvaluationManager evaluation = new EvaluationManager(classifier, test, all);
    }
}
