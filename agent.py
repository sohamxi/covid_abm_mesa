import numpy as np

class InfectionState(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DIED = 3

class JobType(enum):
    GOVERNMENT = 'g'
    BLUE_COLLAR ='l'
    WHITE_COLLAR = 'e'
    UNEMPLOYED = 'u'
    BUSINESS_OWNER = 'b'

# TODO - Need to figure out how to restrict mobility (Lock down, Quarantine)

class Human(Agent):
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # TODO - Age distribution to be taken as per location/ country
        self.age = self.random.normalvariate(20,40)        
        self.state = InfectionState.SUSCEPTIBLE  
        # TODO - Job type to affect Income and thus wealth
        self.jobtype = JobType.WHITE_COLLAR
        self.infection_time = 0

    def move(self):
        """Move the agent"""

        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def status(self):
        """Check infection status"""

        if self.state == InfectionState.INFECTED:     
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate,1-drate])
            if alive == 0:
                self.model.schedule.remove(self)            
            t = self.model.schedule.time-self.infection_time
            if t >= self.recovery_time:          
                self.state = InfectionState.DIED

    def contact(self):
        """Find close contacts and infect"""

        cellmates = self.model.grid.get_cell_list_contents([self.pos])       
        if len(cellmates) > 1:
            for other in cellmates:
                if self.random.random() > self.model.ptrans:
                    continue
                if self.state is InfectionState.INFECTED and other.state is InfectionState.SUSCEPTIBLE:                    
                    other.state = InfectionState.INFECTED
                    other.infection_time = self.model.schedule.time
                    other.recovery_time = self.model.get_recovery_time()

    def step(self):
        self.status()
        self.move()
        self.contact()