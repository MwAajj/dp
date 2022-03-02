package structure;

import lombok.AllArgsConstructor;
import lombok.Getter;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;


import java.io.Serializable;
import java.util.ArrayList;


public interface Tree {

    enum Son {
        NONE,
        LEFT,
        RIGHT,
        BOTH
    }
    void buildTree(Instances data);
    Instances findKNearestNeighbours(Instance instance, int k);
    ArrayList<Attribute> getALlAttributes(Instance instance);
    double[] getDistances();

    @AllArgsConstructor
    @Getter
    class DistInst implements Comparable<Tree.DistInst>, Serializable {
        private Instance instance;
        private double distance;

        @Override
        public int compareTo(DistInst o) {
            return Double.compare(o.distance, this.distance);
        }
    }
}
