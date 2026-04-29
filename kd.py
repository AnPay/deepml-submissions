import numpy as np

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value
	"""
	student_logits=student_logits/temperature
	teacher_logits = teacher_logits/temperature
	teacher_sm =  np.exp(teacher_logits)/(np.sum(np.exp(teacher_logits),axis=-1,keepdims=True))
	student_sm = np.exp(student_logits)/(np.sum(np.exp(student_logits),axis=-1,keepdims=True))
	KL = sum(teacher_sm[i] * np.log(teacher_sm[i]/student_sm[i]) for i in range(len(teacher_sm)))
	L = temperature*temperature * KL
	return L
