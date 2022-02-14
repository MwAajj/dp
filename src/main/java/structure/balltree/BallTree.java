package structure.balltree;

import structure.Tree;
import weka.core.Instance;
import weka.core.Instances;

public class BallTree implements Tree {

    @Override
    public void buildTree(Instances data) {
        System.out.println("Maybe");
    }

    @Override
    public void findKNearestNeighbours(Instance instance, int k) {
    }

}
