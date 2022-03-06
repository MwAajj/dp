package structure.trees.ballTree;

import lombok.Getter;
import lombok.Setter;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

@Getter
@Setter
public class BallTreeNode implements Serializable {
    private Instances instances;
    private BallTreeNode rightSon;
    private BallTreeNode leftSon;
    private Instance centroid;
    private double radius;

    public BallTreeNode(Instance centroid) {
        this.centroid = centroid;
    }

    public void clearInstances() {
        instances = null;
    }

    public boolean isInstances() {
        return instances != null;
    }

    public void addInstance(Instance instance){
        instances.add(instance);
    }

    public boolean isLeaf() {
        return leftSon == null && rightSon == null;
    }
}
