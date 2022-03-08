package classifier.variants;

import weka.core.Instance;

import java.io.Serializable;

public interface Variant extends Serializable {
    double[] distributionForInstance(Instance instance, int m_NumClasses);
    double classifyInstance(Instance instance, int m_NumClasses);
    String getOption();
}
