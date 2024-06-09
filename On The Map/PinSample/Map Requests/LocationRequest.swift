//
//  LocationRequest.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 07/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct LocationRequest: Codable {
    let objectId: String
    
    enum CodingKeys: String, CodingKey {
        case objectId
    }
}
