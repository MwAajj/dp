import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import structure.ballTree.BallTree;
import structure.kdtree.KdTree;
import weka.core.Instances;

public class Source
{
    public static void main(String[] args) throws Exception {
        DatasetManager datasetManager  = new DatasetManager("VsetkoCovid", "VsetkoCovid");



        int k = 2;
        KdTree kdTree = new KdTree(false);
        BallTree ballTree = new BallTree(2);

        //String[] options = {"-K", "3", "-F", "2"};

        InstanceManager manager = new InstanceManager("testData5", 2);


        Instances all = manager.getAll();
        Instances test = manager.getTest();
        Instances train = manager.getTrain();
        ballTree.buildTree(all);
        ballTree.findKNearestNeighbours(all.firstInstance(), 4);

       /*kdTree.buildTree(all);
        kdTree.findKNearestNeighbours(all.firstInstance(), 4);*/

        ballTree.setInstances(all);
        //IBk classifier = new IBk(k);
        MyAlgorithm classifier = new MyAlgorithm(k);
        /*classifier.setNearestNeighbourSearchAlgorithm(kdTree);
        classifier.setOptions(options);*/
        classifier.buildClassifier(all);



        //System.out.println(all.firstInstance());
       // System.out.println(kdTree.nearestNeighbour(all.firstInstance()));

        /*String[] options = {"-K", "3"};
        //IBk classifier = new IBk(3);

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
