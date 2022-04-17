package classifier.variants;

import lombok.AllArgsConstructor;
import lombok.Getter;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

public interface Variant extends Serializable {
    double[] distributionForInstance(Instance instance, int mNumberClasses);
    double classifyInstance(Instance instance, int mNumberClasses);
    String getOption();
    void setOption(String option);
    void sortInstances(Instances instances,  double[] distances);

    @AllArgsConstructor
    @Getter
    class DistInst implements Comparable<DistInst>, Serializable {
        private Instance instance;
        private double distance;

        @Override
        public int compareTo(DistInst o) {
            return Double.compare(o.distance, this.distance);
        }
    }
}
