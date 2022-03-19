package classifier.structure;

import lombok.AllArgsConstructor;
import lombok.Getter;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;


import java.io.Serializable;
import java.util.ArrayList;


public interface Structure {
    void setDistanceFunction(DistanceFunction function);
    void buildStructure(Instances data);
    Instances findKNearestNeighbours(Instance instance, int k);
    ArrayList<Attribute> getALlAttributes(Instance instance);
    double[] getDistances();

    @AllArgsConstructor
    @Getter
    class DistInst implements Comparable<Structure.DistInst>, Serializable {
        private Instance instance;
        private double distance;

        @Override
        public int compareTo(DistInst o) {
            return Double.compare(o.distance, this.distance);
        }
    }
}
